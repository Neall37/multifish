/**
 *
 * Given a list of directory paths that need to be accessed
 * create mount options for the current container engine.
 */
def create_container_options(dirList) {
    def dirs = dirList.unique(false)

    // Define an alternative directory if /data_test does not exist
    def alternativeDir = '/local/workdir/multifish/data_test'

    // Create the alternative directory if it doesn't exist
    new File(alternativeDir).mkdirs()

    // Check for existence of /data_test and replace if necessary
    dirs = dirs.collect { it == '/data_test' && !new File(it).exists() ? alternativeDir : it }

    if (workflow.containerEngine == 'singularity') {
        dirs
        .findAll { it != null && it != '' }
        .inject(params.runtime_opts) { arg, item ->
            if (!arg.contains("-B ${item}")) {
                "${arg} -B ${item}"
            } else {
                arg
            }
        }
    } else if (workflow.containerEngine == 'docker') {
        dirs
        .findAll { it != null && it != '' }
        .inject(params.runtime_opts) { arg, item ->
            if (!arg.contains("-v ${item}:${item}")) {
                "${arg} -v ${item}:${item}"
            } else {
                arg
            }
        }
    } else {
        params.runtime_opts
    }
}
