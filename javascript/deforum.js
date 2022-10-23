function submit_deforum(){
    requestProgress('deforum')

    return []
}

onUiUpdate(function(){
    check_progressbar('deforum', 'deforum_progressbar', 'deforum_progress_span', 'deforum_skip', 'deforum_interrupt', 'deforum_preview', 'deforum_gallery')
})
