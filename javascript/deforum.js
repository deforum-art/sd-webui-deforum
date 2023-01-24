function submit_deforum(){
	// alert('Hello, Deforum!')
    rememberGallerySelection('deforum_gallery')
    showSubmitButtons('deforum', false)

    var id = randomId()
    requestProgress(id, gradioApp().getElementById('deforum_gallery_container'), gradioApp().getElementById('deforum_gallery'), function(){
        showSubmitButtons('deforum', true)
    })

    var res = create_submit_args(arguments)

    res[0] = id
	// res[1] = get_tab_index('deforum')

    return res
}

onUiUpdate(function(){
    check_gallery('deforum_gallery')
})
