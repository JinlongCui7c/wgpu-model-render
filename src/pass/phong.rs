use std::{collections::HashMap, iter, mem};
use bytemuck::{Pod, Zeroable};
use cgmath::{InnerSpace, Rotation3, Zero};
use wgpu::{util::DeviceExt, BindGroupLayout, Device, Queue, Surface};

use crate::{
    camera::{Camera, CameraUniform},
    context::create_render_pipeline,
    instance::{Instance, InstanceRaw},
    model::{self, DrawLight, DrawModel, Model, Vertex},
    node::Node,
    texture,
};

use super::{Pass, UniformPool};

// Global uniform data
// aka camera position and ambient light color
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
    ambient: [f32; 4],
}

// Local uniform data
// aka the individual model's data
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Locals {
    pub position: [f32; 4],
    pub color: [f32; 4],
    pub normal: [f32; 4],
    pub lights: [f32; 4],
}

// Uniform for light data (position + color)
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    pub position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    pub color: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding2: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MyUniforms {
    my_float: f32,
}

pub struct PhongConfig {
    pub max_lights: usize,
    pub ambient: [u32; 4],
    pub wireframe: bool,
}

pub struct PhongPass {
    // Uniforms
    pub global_bind_group_layout: BindGroupLayout,
    pub global_uniform_buffer: wgpu::Buffer,
    pub global_bind_group: wgpu::BindGroup,
    pub local_bind_group_layout: BindGroupLayout,
    // pub local_uniform_buffer: wgpu::Buffer,
    local_bind_groups: HashMap<usize, wgpu::BindGroup>,
    pub uniform_pool: UniformPool,
    // Textures
    pub depth_texture: texture::Texture,
    // Render pipeline
    pub render_pipeline: wgpu::RenderPipeline,
    // Lighting
    pub light_uniform: LightUniform,
    pub light_buffer: wgpu::Buffer,
    // pub light_bind_group: wgpu::BindGroup,
    pub light_render_pipeline: wgpu::RenderPipeline,
    // Camera
    pub camera_uniform: CameraUniform,
    // Instances
    instance_buffers: HashMap<usize, wgpu::Buffer>,
    pub my_uniform_buf:wgpu::Buffer,
}

impl PhongPass {
    pub fn new(
        phong_config: &PhongConfig,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        camera: &Camera,
    ) -> PhongPass {
        // Setup the shader
        // We use specific shaders for each pass to define visual effect
        // and also to have the right shader for the uniforms we pass
        // 在WGPU中，着色器模块是用于运行在GPU上的着色器代码的容器。
        // 它可以包含着色器的源代码，并且由设备（Device）对象创建。
        // 在后续使用这个着色器模块时，可以将其加载到渲染流水线中
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Normal Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shader.wgsl").into()),
        });

        // Setup global uniforms
        // Global bind group layout
        let light_size = mem::size_of::<LightUniform>() as wgpu::BufferAddress;
        let global_size = mem::size_of::<Globals>() as wgpu::BufferAddress;

        // 这段代码创建了一个新的绑定组布局，其中包含了三个绑定组布局条目，用于描述全局、光照和纹理采样器资源的布局规则。
        // 这个绑定组布局将在后续创建绑定组（BindGroup）时使用，以确定如何将资源绑定到着色器中。
        let global_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("[Phong] Globals"),
                entries: &[
                    // 对于全局和光照，我们指定了它们的绑定点（binding）为0和1，
                    // 并设置可见性（visibility）为VERTEX | FRAGMENT，
                    // 表示这些资源在顶点着色器和片段着色器中都可见。
                    // Global uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(global_size),
                        },
                        count: None,
                    },
                    // Lights
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(light_size),
                        },
                        count: None,
                    },

                    // Sampler for textures
                    // 对于纹理采样器，我们指定了它的绑定点为2，并设置可见性为FRAGMENT，表示这个采样器只在片段着色器中可见。
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },

                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        // Global uniform buffer
        // 这段代码创建了一个新的全局 uniform 缓冲区，用于存储全局或场景范围内的数据。
        // 这个缓冲区将用于在渲染过程中传递相机参数、光照信息等全局数据到着色器中。
        let global_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            // 一个可选的标签，用于给这个缓冲区起一个名称，便于调试和识别。
            label: Some("[Phong] Globals"),
            // 缓冲区的大小，即要存储的数据的总字节数
            size: global_size,
            // 缓冲区的使用方式，这里设置为UNIFORM | COPY_DST，
            // 表示这个缓冲区用于存储 uniform 数据，并且可以作为目标进行复制操作。
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            // 示在创建缓冲区时是否映射（映射后可以直接访问缓冲区数据）。在这里设置为false，表示在创建缓冲区时不进行映射。
            mapped_at_creation: false,
        });

        // Create light uniforms and setup buffer for them
        // 通过这种方式，可以轻松地为光源设置位置和颜色等属性，然后将LightUniform结构体实例传递给渲染管线的着色器，以便在渲染过程中使用光照信息。
        let light_uniform = LightUniform {
            // 表示光源的位置。
            position: [2.0, 2.0, 2.0],
            // 这是一个无用的字段，用于填充对齐字节，确保结构体的大小符合对齐要求
            _padding: 0,
            // 表示光源的颜色，即 白色。
            color: [1.0, 1.0, 1.0],
            _padding2: 0,
        };

        // light_buffer缓冲区被成功创建，并且已经存储了light_uniform结构体实例的数据。
        // 之后，可以将light_buffer绑定到渲染管线的Bind Group中，以便在着色器中访问光照数据。
        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("[Phong] Lights"),
            // 这是一个包含光照数据的字节序列。
            // bytemuck::cast_slice(&[light_uniform])将light_uniform转换为字节序列，并将其传递给contents字段。
            contents: bytemuck::cast_slice(&[light_uniform]),
            // 表示该缓冲区既可以用作Uniform缓冲区，也可以作为Copy Destination缓冲区。
            // Uniform缓冲区用于在着色器中存储常量数据，而Copy Destination缓冲区用于在数据复制操作中作为目标缓冲区。
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 定义 Uniforms 旋转角度
        let my_float= 0.0;
        let my_uniforms = MyUniforms {
            my_float: my_float,
        };

        let my_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&my_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // We also need a sampler for our textures
        // 纹理采样器用于在着色器中对纹理进行采样，以获取纹理上特定坐标处的颜色值。
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("[Phong] sampler"),
            // 这两个字段分别用于指定纹理缩小和放大时的采样过滤器。
            // 使用了wgpu::FilterMode::Linear，表示使用线性过滤器进行采样。
            // 线性过滤器会根据纹理像素之间的颜色值插值得到新的颜色值，以实现平滑的纹理渲染效果。
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            // 用于指定纹理的U、V和W轴（纹理坐标轴）的取样地址模式。在这里，使用了默认的取样地址模式，
            // 即wgpu::AddressMode::ClampToEdge，表示超出纹理坐标范围的像素会使用最边缘的纹理像素进行采样。
            ..Default::default()
        });

        // Combine the global uniform, the lights, and the texture sampler into one bind group
        // 创建了一个名为global_bind_group的绑定组（Bind Group），用于将全局的Uniform数据、光照信息和纹理采样器传递给着色器。
        let global_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("[Phong] Globals"),
            // 绑定组布局定义了绑定组中每个条目的类型和绑定点。
            layout: &global_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: global_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                // 取样器
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                // 旋转角度
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: my_uniform_buf.as_entire_binding(),
                },
            ],
        });

        // Setup local uniforms
        // Local bind group layout
        let local_size = mem::size_of::<Locals>() as wgpu::BufferAddress;
        // 创建了名为local_bind_group_layout的绑定组布局，并定义了局部的Uniform数据和网格纹理的绑定点和类型。
        // 这个绑定组布局将用于创建实际的绑定组，以将局部的Uniform数据和网格纹理传递给着色器进行渲染。
        let local_bind_group_layout =
            // 创建绑定组布局时，需要提供一个BindGroupLayoutDescriptor结构体作为参数，该结构体包含了绑定组布局的配置信息。
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("[Phong] Locals"),
                entries: &[
                    // Local uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        // 表示这个Uniform数据在顶点着色器和片段着色器中都可见。
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        // 表示这是一个Uniform缓冲区
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(local_size),
                        },
                        count: None,
                    },
                    // Mesh texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        // 表示这个纹理只在片段着色器中可见
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            });

        // Setup the render pipeline
        // 这个着色器程序布局将在创建渲染管线时使用，以指定图形渲染管线中各个阶段所需的绑定组布局和推送常量范围。
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("[Phong] Pipeline"),
            // 这是一个包含绑定组布局的数组，用于指定图形渲染管线中每个阶段所需的绑定组布局。在这里，指定了两个绑定组布局
            bind_group_layouts: &[&global_bind_group_layout, &local_bind_group_layout],
            // 这是一个用于指定推送常量范围的数组。推送常量是一种在图形渲染管线中用于快速传递少量数据的机制。
            // 在这里，没有指定推送常量范围，所以这个数组为空。
            push_constant_ranges: &[],
        });

        // 创建了一个用于顶点缓冲区的描述数组vertex_buffers。
        // 这个数组用于告诉图形渲染管线如何解释顶点数据，以便正确地渲染模型实例。
        // 1 
        // ModelVertex 用于表示模型顶点数据的属性，比如位置、法线、纹理坐标等。
        // 这里通过调用desc()方法来获取描述符，告诉渲染管线如何解释这个结构体的数据。
        // 2 
        // InstanceRaw是另一个自定义结构体，用于表示模型实例的数据，比如模型的变换矩阵等。
        // 通过调用desc()方法，告诉渲染管线如何解释InstanceRaw结构体的数据。
        let vertex_buffers = [model::ModelVertex::desc(), InstanceRaw::desc()];

        // 创建了一个深度和模板测试状态对象depth_stencil，用于配置深度和模板测试的行为。
        // 深度和模板测试是在渲染管线中用于控制像素是否被绘制的一种技术，用于实现遮挡和阴影效果。
        let depth_stencil = Some(wgpu::DepthStencilState {
            format: texture::Texture::DEPTH_FORMAT,
            // 如果设置为true，则像素的深度值将被写入深度缓冲区，影响后续像素的深度测试。
            depth_write_enabled: true,
            // 指定深度测试的比较函数。这里使用wgpu::CompareFunction::LessEqual，
            // 表示深度测试将通过深度缓冲区的值与新像素的深度值进行比较，
            // 并且只有当新像素的深度值小于等于深度缓冲区的值时，新像素才会被绘制。
            depth_compare: wgpu::CompareFunction::LessEqual,
            // 这里使用了默认值来配置模板测试，表示不对模板缓冲区进行任何操作。
            stencil: Default::default(),
            bias: Default::default(),
        });

        // Enable/disable wireframe mode
        // 这段代码根据phong_config中的wireframe字段的值来选择渲染管线的图元拓扑（Primitive Topology）。
        // 图元拓扑决定了如何组织顶点以形成图形，比如点、线或三角形等。
        let topology = if phong_config.wireframe {
            wgpu::PrimitiveTopology::LineList
        } else {
            wgpu::PrimitiveTopology::TriangleList
        };

        // 定义了渲染管线的图元状态（wgpu::PrimitiveState）。
        // 图元状态用于指定如何渲染图元（例如三角形、线段、点等）以及如何处理剔除（Culling）。
        let primitive = wgpu::PrimitiveState {
            // 设置剔除模式为Back，这表示渲染时会剔除掉背面（反向）的三角形。
            // 在典型的情况下，渲染时只渲染三角形的正面，而背面会被剔除，从而提高渲染性能。
            // Back指的是三角形的顶点按照逆时针方向定义的情况下，背面在视点的反方向。
            cull_mode: Some(wgpu::Face::Back),
            topology,
            ..Default::default()
        };
        
        let multisample = wgpu::MultisampleState {
            ..Default::default()
        };
        let color_format = texture::Texture::DEPTH_FORMAT;

        // 创建渲染管线（wgpu::RenderPipeline），它定义了如何渲染场景中的图元。
        // 并指定了顶点着色器、片段着色器、图元状态、深度/模板状态等渲染管线的各个部分的配置。
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("[Phong] Pipeline"),
            // 包含了着色器绑定组布局信息
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                // 顶点着色器所使用的着色器模块（即之前创建的shader_module）
                module: &shader_module,
                // 顶点着色器的入口点（函数名称）
                entry_point: "vs_main",
                //  顶点缓冲数组，用于告诉渲染管线顶点数据的排列格式。
                buffers: &vertex_buffers,
            },
            primitive,
            depth_stencil: depth_stencil.clone(),
            multisample,

            fragment: Some(wgpu::FragmentState {
                // 片段着色器所使用的着色器模块（即之前创建的shader_module）。
                module: &shader_module,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    // 定义渲染目标的状态。我们只有一个颜色目标（wgpu::ColorTargetState）
                    // 指定了渲染到屏幕时使用的颜色格式、混合状态和写入掩码。
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
        });

        // Create depth texture
        // 字符串，用于为深度纹理指定一个名称（可选，用于调试和标识）。
        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        // Setup camera uniform
        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let light_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Light Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../light.wgsl").into()),
        });

        // 灯光的pipeline
        // 这个渲染管线用于渲染光源，用于计算光照效果而不会对场景中的模型产生实际渲染输出。
        let light_render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("[Phong] Light Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &light_shader,
                    entry_point: "vs_main",
                    // buffers 是一个数组，定义了顶点缓冲区的布局
                    buffers: &[model::ModelVertex::desc()],
                },
                primitive,
                depth_stencil,
                multisample,
                fragment: Some(wgpu::FragmentState {
                    module: &light_shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState {
                            alpha: wgpu::BlendComponent::REPLACE,
                            color: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
            });

        // Create instance buffer
        //  创建了一个新的空哈希映射，用于存储实例缓冲区（instance buffers）
        let instance_buffers = HashMap::new();

        let uniform_pool = UniformPool::new("[Phong] Locals", local_size);

        PhongPass {
            global_bind_group_layout,
            global_uniform_buffer,
            global_bind_group,
            local_bind_group_layout,
            local_bind_groups: Default::default(),
            uniform_pool,
            depth_texture,
            render_pipeline,
            camera_uniform,
            light_uniform,
            light_buffer,
            light_render_pipeline,
            instance_buffers,
            my_uniform_buf
        }
    }
}

impl Pass for PhongPass {
    fn draw(
        &mut self,
        surface: &Surface,
        device: &Device,
        queue: &Queue,
        nodes: &Vec<Node>,
    ) -> Result<(), wgpu::SurfaceError> {
        // 通过调用 surface 对象的 get_current_texture() 方法来获取当前的渲染目标纹理
        let output = surface.get_current_texture()?;
        // 创建了一个纹理视图 view。纹理视图是对纹理的一个视图，它描述了如何访问纹理的内容和格式。
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // 创建了一个命令编码器（Command Encoder）。命令编码器用于将渲染操作组合成一个或多个命令，以便在图形渲染管线上执行。
        // 在 wgpu 中，渲染操作被组织成命令，例如绘制三角形、清除屏幕等。
        // 这些命令将被添加到命令编码器中，并在后续的步骤中提交给渲染队列执行。
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Setup the render pass
        // see: clear color, depth stencil
        // 使用 encoder.begin_render_pass() 方法开始一个渲染通道（Render Pass）。
        // 渲染通道是一系列渲染操作的集合，它们将在一次渲染中执行。
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    // 它表示渲染操作的目标纹理
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Set the clear color during redraw
                        // This is basically a background color applied if an object isn't taking up space
                        // 用于指定在渲染之前和之后如何处理颜色附件。在这里，我们使用 load 字段来设置清除颜色，
                        // 表示在每次渲染通道开始时将颜色附件的内容清除为指定的颜色值。
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                })],
                // Create a depth stencil buffer using the depth texture
                // 创建了一个深度和模板附件 (depth_stencil_attachment)，用于在渲染通道中处理深度测试和模板测试。
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            // Allocate buffers for local uniforms
            // 这个池子用于存储本地 uniform 数据的缓冲区
            // 段代码用于检查当前池子中的缓冲区数量是否足够，
            // 如果不够就根据节点的数量分配更多的缓冲区，以确保每个节点都有一个本地 uniform 缓冲区。
            // 这样可以为每个节点存储独立的本地 uniform 数据，并在渲染时使用它们。
            if (self.uniform_pool.buffers.len() < nodes.len()) {
                self.uniform_pool.alloc_buffers(nodes.len(), &device);
            }

            // Loop over the nodes/models in a scene and setup the specific models
            // local uniform bind group and instance buffers to send to shader
            // This is separate loop from the render because of Rust ownership
            // (can prob wrap in block instead to limit mutable use)
            // 循环遍历场景中的节点/模型并设置特定的模型
            // 本地统一绑定组和实例缓冲区发送到shader
            // 由于Rust的所有权，这是从渲染中分离出来的循环 (可以在block中探测wrap来限制可变的使用)
            // 在这段代码中，你正在为场景中的每个模型（或节点）创建本地 uniform 数据的绑定组（BindGroup）和实例缓冲区（Buffer）
            let mut model_index = 0;
            for node in nodes {
                // 这里获取了池子中索引为 model_index 的本地 uniform 缓冲区，用于存储与当前节点相关的本地 uniform 数据。
                let local_buffer = &self.uniform_pool.buffers[model_index];

                // We create a bind group for each model's local uniform data
                // and store it in a hash map to look up later
                // 于创建或获取与模型索引 model_index 相关联的本地 uniform 绑定组。
                // 如果哈希表中已经存在这个索引对应的绑定组，则直接返回该绑定组；如果不存在，则创建一个新的绑定组并存储在哈希表中。
                // 绑定组是用于在着色器中访问本地 uniform 数据的方式。
                self.local_bind_groups
                    .entry(model_index)
                    .or_insert_with(|| {
                        device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("[Phong] Locals"),
                            layout: &self.local_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: local_buffer.as_entire_binding(),
                                },
                                // 这里获取了模型的第一个材质的漫反射贴图（diffuse texture）的视图（TextureView）。
                                // 这个视图将被绑定到绑定组中的第二个绑定点（binding 1），用于在着色器中访问该贴图。
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::TextureView(
                                        // 绑定纹理贴图
                                        &node.model.materials[0].diffuse_texture.view,
                                    ),
                                },
                            ],
                        })
                    });

                // Setup instance buffer for the model
                // similar process as above using HashMap
                // 这是另一个哈希表操作，用于创建或获取与模型索引 model_index 相关联的实例缓冲区。
                // 实例缓冲区是用于存储模型的实例化数据（比如变换矩阵）的缓冲区，允许多个实例共享相同的顶点数据，
                // 并通过实例化渲染多个相同的模型。
                self.instance_buffers.entry(model_index).or_insert_with(|| {
                    // We condense the matrix properties into a flat array (aka "raw data")
                    // (which is how buffers work - so we can "stride" over chunks)
                    let instance_data = node
                        .instances
                        .iter()
                        .map(Instance::to_raw)
                        .collect::<Vec<_>>();
                    // Create the instance buffer with our data
                    let instance_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Instance Buffer"),
                            contents: bytemuck::cast_slice(&instance_data),
                            usage: wgpu::BufferUsages::VERTEX,
                        });

                    instance_buffer
                });

                model_index += 1;
            }

            // Setup lighting pipeline
            // 用于设置渲染通道（RenderPass）的渲染管线（RenderPipeline）为 self.light_render_pipeline，以准备进行光源渲染。
            render_pass.set_pipeline(&self.light_render_pipeline);

            // Draw/calculate the lighting on models
            // 这段代码调用了 render_pass 对象的 draw_light_model 方法，用于在渲染通道中绘制光源模型。
            /*
            render_pass.draw_light_model(
                // 表示要绘制的模型对象
                &nodes[0].model,
                // 它包含了全局的着色器绑定（uniform buffer、光源信息等）。
                // 这个绑定组会在绘制时与着色器进行绑定，使着色器能够访问全局的数据。
                &self.global_bind_group,
                // 表示局部绑定组（BindGroup）。
                // 这里使用 self.local_bind_groups.get(&1) 获取了一个局部绑定组，它与光源模型相关的局部着色器绑定。
                &self
                    .local_bind_groups
                    .get(&1)
                    .expect("No local bind group found for lighting"),
            );
            */

            // Setup render pipeline
            // 将渲染通道的渲染管线设置为 self.render_pipeline。这个渲染管线（render_pipeline）是预先创建好的用于渲染的管线对象，
            // 它包含了渲染所需的顶点着色器、片元着色器、深度测试等配置。
            render_pass.set_pipeline(&self.render_pipeline);
            // 将全局绑定组（global_bind_group）绑定到渲染通道的指定槽（slot）上。
            // 使用槽索引 0 来绑定全局绑定组。第一个参数 0 表示槽索引，
            // 第二个参数 &self.global_bind_group 是要绑定的全局绑定组，
            // 第三个参数 &[] 表示动态偏移量（dynamic offsets），这里为空数组表示没有动态偏移量。
            render_pass.set_bind_group(0, &self.global_bind_group, &[]);

            // Render/draw all nodes/models
            // We reset index here to use again
            model_index = 0;
            for node in nodes {
                // Set the instance buffer unique to the model
                // 将实例缓冲（self.instance_buffers[&model_index]）绑定到顶点缓冲槽索引 1 上。
                // 实例缓冲是一个包含了多个实例属性的缓冲区，每个实例属性包含了模型的位置、旋转、缩放等信息。
                // 通过将实例缓冲绑定到顶点缓冲槽索引 1 上，使得顶点着色器能够使用实例属性对模型进行实例化渲染。
                render_pass.set_vertex_buffer(1, self.instance_buffers[&model_index].slice(..));

                // Draw all the model instances
                render_pass.draw_model_instanced(
                    &node.model,
                    0..*&node.instances.len() as u32,
                    &self.local_bind_groups[&model_index],
                );

                model_index += 1;
            }
        }

        queue.submit(Some(encoder.finish()));
        output.present();

        // Since the WGPU breaks return with a Result and error
        // we need to return an `Ok` enum
        Ok(())
    }
}
