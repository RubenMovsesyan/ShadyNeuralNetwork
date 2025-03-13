pub enum Bbt {
    Uniform,
    Storage,
}

#[macro_export]
macro_rules! create_buffer_bind_group {
    ( $device:expr, $label:expr, $( ($binding:expr, $buffer:expr, $type:expr, $read_only:expr) ),* ) => {
        {
            use crate::layer::bind_group_macro::Bbt;
            let mut layout_entries = Vec::new();
            let mut bind_group_entries = Vec::new();
            $(
                let buffer_binding_type = match $type {
                    Bbt::Uniform => BufferBindingType::Uniform,
                    Bbt::Storage => BufferBindingType::Storage { read_only: $read_only }
                };

                layout_entries.push(BindGroupLayoutEntry {
                    binding: $binding,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });

                bind_group_entries.push(BindGroupEntry {
                    binding: $binding,
                    resource: $buffer.as_entire_binding(),
                });
            )*

            let bind_group_layout = $device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some(&format!("{} Layout", $label)),
                entries: &layout_entries,
            });

            let bind_group = $device.create_bind_group(&BindGroupDescriptor {
                label: Some($label),
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            });

            (bind_group_layout, bind_group)
        }
    };
}
