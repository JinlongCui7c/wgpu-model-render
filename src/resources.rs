use std::io::{BufReader, Cursor};

use cfg_if::cfg_if;
use gltf::Gltf;
use wgpu::util::DeviceExt;

use crate::{
    model::{self, AnimationClip, Keyframes, ModelVertex},
    texture,
};

#[cfg(target_arch = "wasm32")]
fn format_url(file_name: &str) -> reqwest::Url {
    let window = web_sys::window().unwrap();
    let location = window.location();
    let base = reqwest::Url::parse(&format!(
        "{}/{}/",
        location.origin().unwrap(),
        option_env!("RES_PATH").unwrap_or("assets"),
    ))
    .unwrap();
    base.join(file_name).unwrap()
}

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            log::warn!("Load model on web");

            let url = format_url(file_name);
            let txt = reqwest::get(url)
                .await?
                .text()
                .await?;

            log::warn!("{}", txt);

        } else {
            let path = std::path::Path::new("assets")
                .join(file_name);
            let txt = std::fs::read_to_string(path)?;
        }
    }

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            let url = format_url(file_name);
            let data = reqwest::get(url)
                .await?
                .bytes()
                .await?
                .to_vec();
        } else {
            let path = std::path::Path::new("assets")
                .join(file_name);
            let data = std::fs::read(path)?;
        }
    }

    Ok(data)
}

pub async fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub async fn load_model_gltf(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<model::Model> {
    let gltf_text = load_string(file_name).await?;
    let gltf_cursor = Cursor::new(gltf_text);
    let gltf_reader = BufReader::new(gltf_cursor);
    let gltf = Gltf::from_reader(gltf_reader)?;

    // Load buffers
    let mut buffer_data = Vec::new();
    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {

            }
            gltf::buffer::Source::Uri(uri) => {
                let bin = load_binary(uri).await?;
                buffer_data.push(bin);
            }
        }
    }

    // Load animations
    let mut animation_clips = Vec::new();
    for animation in gltf.animations() {
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(&buffer_data[buffer.index()]));
            let timestamps = if let Some(inputs) = reader.read_inputs() {
                match inputs {
                    gltf::accessor::Iter::Standard(times) => {
                        let times: Vec<f32> = times.collect();
                        println!("Time: {}", times.len());
                        dbg!(&times);
                        times
                    }
                    gltf::accessor::Iter::Sparse(_) => {
                        println!("Sparse keyframes not supported");
                        let times: Vec<f32> = Vec::new();
                        times
                    }
                }
            } else {
                println!("We got problems");
                let times: Vec<f32> = Vec::new();
                times
            };

            let keyframes = if let Some(outputs) = reader.read_outputs() {
                match outputs {
                    gltf::animation::util::ReadOutputs::Translations(translation) => {
                        let translation_vec = translation.map(|tr| {
                            // println!("Translation:");
                            dbg!(&tr);
                            let vector: Vec<f32> = tr.into();
                            vector
                        }).collect();
                        Keyframes::Translation(translation_vec)
                    },
                    other => {
                        Keyframes::Other
                    }
                }
            } else {
                println!("We got problems");
                Keyframes::Other
            };

            animation_clips.push(AnimationClip {
                name: animation.name().unwrap_or("Default").to_string(),
                keyframes,
                timestamps,
            })
        }
    }

    // Load materials
    let mut materials = Vec::new();
    for material in gltf.materials() {
        println!("Looping thru materials");
        // 从当前材质中获取PBR（Physically Based Rendering）金属粗糙度信息。
        let pbr = material.pbr_metallic_roughness();
        
        // 获取基础颜色贴图的信息。基础颜色贴图用于存储物体表面的颜色信息。
        let base_color_texture = &pbr.base_color_texture();

        // 获取基础颜色贴图的源数据，
        // 可能是一个gltf::image::Source::View或者gltf::image::Source::Uri。
        // 这里使用了map()方法来转换贴图的源数据。
        let texture_source = &pbr
            .base_color_texture()
            .map(|tex| {
                // println!("Grabbing diffuse tex");
                // dbg!(&tex.texture().source());
                tex.texture().source().source()
            })
            .expect("texture");
        
        // 根据贴图源数据的类型，分别处理gltf::image::Source::View和gltf::image::Source::Uri两种情况：
        // 如果贴图源是gltf::image::Source::View，则从指定的view中读取贴图数据，
        // 并使用texture::Texture::from_bytes()方法将贴图数据加载为纹理。
        // 然后将包含材质名称和贴图的结构体model::Material添加到材质数组中。
        match texture_source {
            gltf::image::Source::View { view, mime_type } => {
                let diffuse_texture = texture::Texture::from_bytes(
                    device,
                    queue,
                    &buffer_data[view.buffer().index()],
                    file_name,
                )
                .expect("Couldn't load diffuse");

                materials.push(model::Material {
                    name: material.name().unwrap_or("Default Material").to_string(),
                    diffuse_texture,
                });
            }
            
            gltf::image::Source::Uri { uri, mime_type } => {
                let diffuse_texture = load_texture(uri, device, queue).await?;

                materials.push(model::Material {
                    name: material.name().unwrap_or("Default Material").to_string(),
                    diffuse_texture,
                });
            }
        };
    }

    let mut meshes = Vec::new();

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            println!("Node {}", node.index());
            // dbg!(node);

            let mesh = node.mesh().expect("Got mesh");
            let primitives = mesh.primitives();
            primitives.for_each(|primitive| {
                // dbg!(primitive);

                let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

                let mut vertices = Vec::new();
                if let Some(vertex_attribute) = reader.read_positions() {
                    vertex_attribute.for_each(|vertex| {
                        // dbg!(vertex);
                        vertices.push(ModelVertex {
                            position: vertex,
                            tex_coords: Default::default(),
                            normal: Default::default(),
                        })
                    });
                }
                if let Some(normal_attribute) = reader.read_normals() {
                    let mut normal_index = 0;
                    normal_attribute.for_each(|normal| {
                        // dbg!(normal);
                        vertices[normal_index].normal = normal;

                        normal_index += 1;
                    });
                }
                // 纹理坐标
                if let Some(tex_coord_attribute) = reader.read_tex_coords(0).map(|v| v.into_f32()) {
                    let mut tex_coord_index = 0;
                    tex_coord_attribute.for_each(|tex_coord| {
                        // dbg!(tex_coord);
                        vertices[tex_coord_index].tex_coords = tex_coord;

                        tex_coord_index += 1;
                    });
                }

                let mut indices = Vec::new();
                if let Some(indices_raw) = reader.read_indices() {
                    // dbg!(indices_raw);
                    indices.append(&mut indices_raw.into_u32().collect::<Vec<u32>>());
                }

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Vertex Buffer", file_name)),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("{:?} Index Buffer", file_name)),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

                meshes.push(model::Mesh {
                    name: file_name.to_string(),
                    vertex_buffer,
                    index_buffer,
                    num_elements: indices.len() as u32,
                    // material: m.mesh.material_id.unwrap_or(0),
                    material: 0,
                });
            });
        }
    }

    Ok(model::Model {
        meshes,
        materials,
        animations: animation_clips,
    })
}
