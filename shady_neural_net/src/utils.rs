use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, Device, Maintain, MapMode};

#[allow(dead_code)]
pub fn read_buffer(
    buffer: &Buffer,
    buffer_size: u64,
    device: &Device,
    encoder: &mut CommandEncoder,
) -> Buffer {
    let read_buffer = device.create_buffer(&BufferDescriptor {
        label: Some("read buffer"),
        size: buffer_size,
        mapped_at_creation: false,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    });

    encoder.copy_buffer_to_buffer(buffer, 0, &read_buffer, 0, buffer_size);

    read_buffer
}

#[allow(dead_code)]
pub fn print_buffer(buffer: &Buffer, device: &Device, name: &str) {
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device.poll(Maintain::Wait);

    let data = slice.get_mapped_range();

    let new_slice: &[f32] = bytemuck::cast_slice(&data);

    println!("{}: {:#?}", name, new_slice.to_vec());

    drop(data);
    buffer.unmap();
}

#[allow(dead_code)]
pub fn get_buffer(buffer: &Buffer, device: &Device) -> Vec<f32> {
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device.poll(Maintain::Wait);

    let data = slice.get_mapped_range();

    let new_slice: &[f32] = bytemuck::cast_slice(&data);

    let output = new_slice.to_vec();

    drop(data);
    buffer.unmap();

    output
}
