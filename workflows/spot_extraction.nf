include {
    cut_tiles;
    airlocalize;
    merge_points;
} from '../processes/spot_extraction' addParams(lsf_opts: params.lsf_opts, 
                                                spotextraction_container: params.spotextraction_container)

include {
    index_channel;
} from '../utils/utils'

workflow spot_extraction {
    take:
    input_dir
    output_dir
    channels
    scale
    xy_stride
    xy_overlap
    z_stride
    z_overlap
    dapi_channel
    dapi_correction_channels
    per_channel_air_localize_params

    main:
    def tile_cut_res = cut_tiles(
        input_dir,
        channels[0],
        scale,
        output_dir,
        xy_stride,
        xy_overlap,
        z_stride,
        z_overlap
    )

    def indexed_tile_cut_input = index_channel(tile_cut_res[0])
    def tiles_with_inputs = indexed_tile_cut_input | join(index_channel(tile_cut_res[1])) | flatMap {
        def tile_input = it[1]
        it[2].tokenize(' ').collect {
            [ tile_input, it ]
        }
    }

    def airlocalize_inputs = tiles_with_inputs | combine(channels) | map {
        def tile_input = it[0]
        def tile_dir = it[1]
        def ch = it[2]
        def dapi_correction = dapi_correction_channels.contains(ch)
                ? "/${dapi_channel}/${scale}"
                : ''
        // return a tuple with all required arguments for airlocalize
        def airlocalize_args = [
            tile_input,
            ch,
            scale,
            "${tile_dir}/coords.txt",
            per_channel_air_localize_params[ch],
            tile_dir,
            "_${ch}.txt",
            dapi_correction
        ]
        println "Create airlocalize args: ${airlocalize_args}"
        return airlocalize_args
    }

    def airlocalize_results = airlocalize(
        airlocalize_inputs.map { it[0] },
        airlocalize_inputs.map { it[1] },
        airlocalize_inputs.map { it[2] },
        airlocalize_inputs.map { it[3] },
        airlocalize_inputs.map { it[4] },
        airlocalize_inputs.map { it[5] },
        airlocalize_inputs.map { it[6] },
        airlocalize_inputs.map { it[7] }
    ) | groupTuple(by: [0, 1])

    def merge_points_inputs = airlocalize_results | map {
        def merge_points_args = it
        println "Create merge point args: ${merge_points_args}"
        return merge_points_args
    }

    def merge_points_results = merge_points(
        merge_points_inputs.map { it[0] }
        merge_points_inputs.map { it[1] }
        merge_points_inputs.map { it[2] }
        merge_points_inputs.map { it[3] }
        merge_points_inputs.map { it[4] }
        merge_points_inputs.map { it[5] }
        merge_points_inputs.map { it[6] }
    )

    // | map {
    //     ch = it[0]
    //     args = it[2]
    //     merge_points_args = [
    //         args.data_dir,
    //         ch,
    //         args.scale,
    //         args.spot_extraction_output_dir,
    //         args.xy_overlap,
    //         args.z_overlap,
    //         args.spot_extraction_output_dir
    //     ]
    //     println "Prepare merge points args: ${merge_points_args}"
    //     return merge_points_args
    // } \
    // | merge_points \
    // | set { done }

    emit:
    done = merge_points_results
}
