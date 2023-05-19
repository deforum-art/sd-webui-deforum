/*
 * 'Deforum' plugin for Automatic1111's Stable Diffusion WebUI.
 * Copyright (C) 2023 Artem Khrapov (kabachuha) and Deforum team listed in AUTHORS.md
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact the dev team: https://discord.gg/deforum
 */

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