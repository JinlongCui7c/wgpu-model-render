use std::{iter, time::Instant};
use bytemuck::{Pod, Zeroable};
use cgmath::prelude::*;
use context::GraphicsContext;
use node::Node;
use pass::{phong::PhongPass, Pass};
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::*,
    event_loop::{ControlFlow, EventLoop},
};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MyUniforms {
    my_float: f32,
}

mod camera;
mod context;
mod instance;
mod model;
mod node;
mod pass;
mod primitives;
mod resources;
mod texture;
mod window;
use crate::{
    camera::{Camera, CameraController, CameraUniform},
    context::create_render_pipeline,
    model::Keyframes,
    pass::phong::{Locals, PhongConfig},
    primitives::{sphere::generate_sphere, PrimitiveMesh},
    window::Window,
};
use crate::{
    instance::{Instance, InstanceRaw},
    window::WindowEvents,
};
use model::{DrawLight, DrawModel, Vertex};


struct State {
    ctx: GraphicsContext,
    pass: PhongPass,
    // Window size
    size: winit::dpi::PhysicalSize<u32>,
    // Clear color for mouse interactions
    clear_color: wgpu::Color,
    // Camera
    camera: Camera,
    camera_controller: CameraController,
    // The 3D models in the scene (as Nodes)
    nodes: Vec<Node>,
    // Animation
    time: Instant,
    my_float: f32,
}

impl State {
    // Initialize the state
    // State::new 函数：它是应用程序的初始化函数。
    // 在这个函数中，我们创建图形上下文，初始化相机、渲染通道和模型，并设置一些初始状态。
    async fn new(window: &Window) -> Self {
        // Save the window size for use later
        let size = window.window.inner_size();

        // Initialize the graphic context
        let ctx = GraphicsContext::new(&window).await;

        // Setup the camera and it's initial position
        // 相机的初始参数
        let mut camera = Camera {
            // 摄像机的位置，是一个三维向量，表示摄像机在世界空间中的坐标。
            eye: (0.0, 5.0, -10.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            // 摄像机的上方向，也是一个三维向量，用于指定摄像机的上方朝向。
            up: cgmath::Vector3::unit_y(),
            // 摄像机的视角宽高比，通常使用窗口宽度除以高度得到。
            aspect: ctx.config.width as f32 / ctx.config.height as f32,
            // 摄像机的垂直视野角度，通常以角度为单位，表示摄像机从上到下的可视范围。
            fovy: 60.0,
            // 摄像机的近裁剪面，表示距离摄像机多近的物体将被裁剪掉。
            znear: 0.1,
            // 摄像机的远裁剪面，表示距离摄像机多远的物体将被裁剪掉。
            zfar: 100.0,
        };

        // 相机向前移动
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        camera.eye = camera.eye+forward_norm *6.0;

        let camera_controller = CameraController::new(0.2);

        // Initialize the pass
        let pass_config = PhongConfig {
            max_lights: 1,
            ambient: Default::default(),
            wireframe: false,
        };
        let pass = PhongPass::new(&pass_config, &ctx.device, &ctx.queue, &ctx.config, &camera);

        // Create the 3D objects!
        // Load 3D model from disk or as a HTTP request (for web support)
        log::warn!("Load model");

        let gltf_model =
            resources::load_model_gltf("FlightHelmet.gltf", &ctx.device, &ctx.queue)
                .await
                .expect("Couldn't load model. Maybe path is wrong?");

        let plane_primitive = PrimitiveMesh::new(
            &ctx.device,
            &ctx.queue,
            &primitives::plane::plane_vertices(0.5),
            &primitives::plane::plane_indices(),
        )
        .await;

        // Create instances for each object with locational data (position + rotation)
        // Renderer currently defaults to using instances. Want one object? Pass a Vec of 1 instance.

        let plane_primitive_instances = vec![Instance {
            position: cgmath::Vector3 {
                x: -3.0,
                y: 3.0,
                z: -3.0,
            },
            rotation: cgmath::Quaternion::from_axis_angle(
                cgmath::Vector3::unit_z(),
                cgmath::Deg(0.0),
            ),
        }];


        let plane_primitive_node = Node {
            parent: 0,
            locals: Locals {
                // x 控制左右，y控制上下
                position: [3.0, -2.0, 0.0, 0.0],
                color: [0.0, 0.0, 1.0, 1.0],
                normal: [0.0, 0.0, 0.0, 0.0],
                lights: [0.0, 0.0, 0.0, 0.0],
            },
            model: gltf_model,
            instances: plane_primitive_instances,
        };

        // Put all our nodes into an Vector to loop over later
        let nodes = vec![
            plane_primitive_node,
        ];

        // Clear color used for mouse input interaction
        let clear_color = wgpu::Color::BLACK;

        let time = Instant::now();

        let my_float=0.0;
        Self {
            ctx,
            pass,
            clear_color,
            size,
            camera,
            camera_controller,
            nodes,
            time,
            my_float
        }
    }

    // Keeps state in sync with window size when changed
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.ctx.config.width = new_size.width;
            self.ctx.config.height = new_size.height;
            self.ctx
                .surface
                .configure(&self.ctx.device, &self.ctx.config);
            // Make sure to current window size to depth texture - required for calc
            self.pass.depth_texture = texture::Texture::create_depth_texture(
                &self.ctx.device,
                &self.ctx.config,
                "depth_texture",
            );
        }
    }

    // Handle input using WindowEvent 
    // 它们处理键盘和鼠标输入，并将事件传递给相机控制器。
    pub fn keyboard(&mut self, state: ElementState, keycode: &VirtualKeyCode) -> bool {
        // Send any input to camera controller
        self.camera_controller.process_events(&state, &keycode)
    }

    pub fn mouse_moved(&mut self, position: &PhysicalPosition<f64>) {
        self.camera_controller
            .process_mouse_moved(&position, &self.size);
    }

    pub fn mouse_input(
        &mut self,
        device_id: &DeviceId,
        state: &ElementState,
        button: &MouseButton,
    ) {
        self.camera_controller
            .process_mouse_input(device_id, state, button);
    }

    // 方法 update，它在应用程序的每次循环迭代中被调用，用于更新应用程序的状态和场景。
    // 它在每一帧更新应用程序的状态，例如相机位置、光源位置和模型的动画。
    fn update(&mut self) {
        // 同步应用程序中的相机状态：
        // 它通过调用 self.camera_controller.update_camera 方法来更新相机的位置和方向，然后更新渲染管道的相机参数。
        // Sync local app state with camera
        // 将应用程序的相机与相机控制器同步，以确保相机位置和方向的正确性。
        self.camera_controller.update_camera(&mut self.camera);
        // 它调用渲染管道的 camera_uniform（相机 Uniform 数据）的 update_view_proj 方法，
        // 将相机的视图投影矩阵更新到相机 Uniform 数据中。
        // 这个视图投影矩阵将在渲染管道中用于将场景从世界坐标系变换到相机视图坐标系。
        self.pass.camera_uniform.update_view_proj(&self.camera);
        // 它将更新后的相机 Uniform 数据写入到渲染队列中，以便后续在 GPU 上使用。
        // 这里使用了 bytemuck::cast_slice 函数将相机 Uniform 数据转换为字节切片，
        // 并将其写入 self.pass.global_uniform_buffer 缓冲区的偏移量 0 处。
        self.ctx.queue.write_buffer(
            &self.pass.global_uniform_buffer,
            0,
            bytemuck::cast_slice(&[self.pass.camera_uniform]),
        );

        // 场景的旋转
        // 更新实体的旋转矩阵
        self.my_float += 0.2;
        let my_uniforms = MyUniforms {
            my_float: self.my_float,
        };
        self.ctx.queue.write_buffer(&self.pass.my_uniform_buf, 0, bytemuck::bytes_of(&my_uniforms));

        // 更新光源：
        // 它通过对光源的旋转来模拟光源的移动，这里使用了四元数旋转来进行计算。然后将更新后的光源信息写入光源缓冲区。
        // Update the light
        // 它将当前光源位置（存储在 self.pass.light_uniform.position）转换为 cgmath::Vector3 类型，以便稍后进行旋转操作。
        let old_position: cgmath::Vector3<_> = self.pass.light_uniform.position.into();

        // 这是一个旋转操作，它将光源的位置向量绕 Y 轴旋转 1 度。
        // 这个操作的目的是改变光源的位置，以产生动态效果。
        // 旋转操作使用 cgmath::Quaternion::from_axis_angle 函数创建一个绕轴旋转的四元数，
        // 并将其应用于原始的光源位置向量，最终将结果转换回 cgmath::Vector3 类型。
        self.pass.light_uniform.position =
            (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
                * old_position)
                .into();

        // 与更新相机 Uniform 数据类似，这里将更新后的光源 Uniform 数据写入渲染队列。
        // 使用 bytemuck::cast_slice 函数将光源 Uniform 数据转换为字节切片，
        // 并将其写入 self.pass.light_buffer 缓冲区的偏移量 0 处。
        self.ctx.queue.write_buffer(
            &self.pass.light_buffer,
            0,
            bytemuck::cast_slice(&[self.pass.light_uniform]),
        );


        // 这段代码用于在场景中更新模型节点的本地状态，以实现动画效果。它按照时间流逝的方式，
        // 根据动画关键帧的时间戳更新节点的本地状态，从而让模型实现动画效果。

        // 更新本地 Uniform 数据：
        // 这一部分代码用于处理模型的动画和局部坐标等信息。
        // 它遍历应用程序中的每个节点（即模型），检查是否有动画并根据当前时间更新动画的关键帧。
        // 如果有动画，它会找到当前关键帧，并根据关键帧更新模型的局部坐标信息。
        // 然后，它通过调用 self.pass.uniform_pool.update_uniform 方法，
        // 将更新后的局部坐标信息写入相应的 Uniform 缓冲区。
        
        // Update local uniforms
        // 获取从应用程序启动以来的时间流逝（以秒为单位），这将用作动画的时间戳。
        let current_time = &self.time.elapsed().as_secs_f32();
        let mut node_index = 0;
        // 遍历每个模型节点，其中 &mut self.nodes 表示可变引用。
        for node in &mut self.nodes {
            // Play animations
            // 检查当前节点是否有动画。如果节点有动画，就进入动画更新的逻辑。
            if node.model.animations.len() > 0 {
                // Loop through all animations
                // TODO: Ideally we'd play a certain animation by name - we assume first one for now
                let mut current_keyframe_index = 0;
                // 嵌套循环遍历节点的动画，并找到与当前时间匹配的关键帧索引 
                for animation in &node.model.animations {
                    for timestamp in &animation.timestamps {
                        if timestamp > current_time {
                            break;
                        }
                        if &current_keyframe_index < &(&animation.timestamps.len() - 1) {
                            current_keyframe_index += 1;
                        }
                    }
                }

                // Update locals with current animation
                // 假设当前我们只考虑节点的第一个动画，获取该动画的关键帧信息。
                let current_animation = &node.model.animations[0].keyframes;
                let mut current_frame: Option<&Vec<f32>> = None;
                // 匹配当前动画的类型，这里我们假设只考虑平移类型的动画（Keyframes::Translation）。
                match current_animation {
                    Keyframes::Translation(frames) => {
                        current_frame = Some(&frames[current_keyframe_index])
                    }
                    Keyframes::Other => (),
                }

                // 如果当前帧数据存在（current_frame.is_some()），
                // 则将当前帧的平移信息更新到节点的本地状态中，以实现模型的动画平移效果。
                if current_frame.is_some() {
                    let current_frame = current_frame.unwrap();

                    node.locals.position = [
                        current_frame[0],
                        current_frame[1],
                        current_frame[2],
                        node.locals.position[3],
                    ];
                }
            }

            &self
                .pass
                .uniform_pool
                .update_uniform(node_index, node.locals, &self.ctx.queue);
            node_index += 1;
        }
    }

    // Primary render flow
    // 它进行实际的渲染。在这里，它调用 self.pass.draw 方法来执行渲染通道的绘制操作。
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        match self.pass.draw(
            &self.ctx.surface,
            &self.ctx.device,
            &self.ctx.queue,
            &self.nodes,
        ) {
            Err(err) => println!("Error in rendering"),
            Ok(_) => (),
        }

        Ok(())
    }
}

// 这是应用程序的入口点。它创建一个 Window 对象，然后通过 Window::run 方法进入主循环。
// 在主循环中，它处理窗口事件并调用相应的状态方法。
pub async fn run() {
    let window = Window::new();

    // State::new uses async code, so we're going to wait for it to finish
    let mut app = State::new(&window).await;

    // @TODO: Wire up state methods again (like render)
    // 使用 winit 库来创建一个窗口并运行事件循环，用于处理窗口事件，如调整大小、键盘输入和鼠标输入。
    window.run(move |event| match event {
        // 当窗口大小发生变化时，调用 app.resize 方法，将新的宽度和高度传递给应用程序的 resize 方法。
        WindowEvents::Resized { width, height } => {
            app.resize(winit::dpi::PhysicalSize { width, height });
        }

        // 当需要绘制窗口内容时，首先调用 app.update 方法，用于更新应用程序的状态。
        // 然后尝试调用 app.render 方法来渲染场景。
        WindowEvents::Draw => {
            app.update();
            match app.render() {
                Err(err) => println!("Error in rendering"),
                Ok(_) => (),
            }
        }

        // 当有键盘事件时，调用 app.keyboard 方法，传递键盘事件的状态和虚拟键码给应用程序的 keyboard 方法。
        WindowEvents::Keyboard {
            state,
            virtual_keycode,
        } => {
            app.keyboard(state, virtual_keycode);
        }

        // 当鼠标移动事件发生时，调用 app.mouse_moved 方法，传递鼠标位置给应用程序的 mouse_moved 方法。
        WindowEvents::MouseMoved { position } => {
            app.mouse_moved(position);
        }

        // 当鼠标输入事件发生时，调用 app.mouse_input 方法，传递设备 ID、状态和按钮信息给应用程序的 mouse_input 方法。
        WindowEvents::MouseInput {
            device_id,
            state,
            button,
        } => {
            app.mouse_input(device_id, state, button);
        }

    });
}
