process predict {
    label 'withGPU'
    label 'large'

    container { params.segmentation_container }
    cpus { params.segmentation_cpus }
    memory { params.segmentation_memory }

    input:
    val(image_path)
    val(ch)
    val(scale)
    val(model_path)
    val(output_path)

    output:
    tuple val(image_path), val(output_path)

    script:
    def output_file = file(output_path)

    args_list = [
        '-i', image_path,
        '-m', model_path,
        '-c', ch,
        '-s', scale,
        '-o', output_path,
        '--big', params.segmentation_big,
        '--n_workers', params.segmentation_n_workers,
        '--batch_size', params.segmentation_batch_size,
        '--threads_per_worker', params.segmentation_threads_per_worker,
        '--num_blocks', params.segmentation_num_blocks,
        '--memory_per_worker', params.segmentation_memory_per_worker
    ]
    args = args_list.collect { it.toString() }.join(' ')
    """
    mkdir -p ${output_file.parent}
    echo "python /app/segmentation/scripts/starfinity_prediction.py ${args}"
    python /app/segmentation/scripts/starfinity_prediction.py ${args}
    """
}
