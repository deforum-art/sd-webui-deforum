function submit_deforum(){
    rememberGallerySelection('deforum_gallery')
    showSubmitButtons('deforum', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('deforum_gallery_container'), gradioApp().getElementById('deforum_gallery'), function(){
        showSubmitButtons('deforum', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id

    return res
}