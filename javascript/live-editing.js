
function live_editing()
{
  return "Hello, Live Editing!"
}
//on ready
console.log("Live Editing 0!");
onUiLoaded(function () {
  const targetNode = gradioApp().querySelector("#deforum_gallery_container");
  console.log(targetNode);
  setTimeout(function () {  const targetNode = gradioApp().querySelector("#deforum_gallery_container");
  console.log(targetNode);}, 5000);
  // Create an observer instance
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      if (mutation.type === 'childList') {
        // Get all added nodes with the .livePreview class
        const addedNodes = mutation.addedNodes;
        for (let i = 0; i < addedNodes.length; i++) {
          node = addedNodes[i];
          if (node.classList && node.classList.contains('livePreview')) {
            // Perform any action you want on the node
            console.log('New live preview node added:', node);
            node.onclick = function (event) {
              const image = event.target
              //get the clicked position, relative to the image resolution

              const containedSize = getContainedSize(image);
              const leftOffset = (image.width - containedSize[0]) / 2;
              const topOffset = (image.height - containedSize[1]) / 2;
              var x = (event.offsetX - leftOffset) / containedSize[0] ;
              var y = (event.offsetY - topOffset) / containedSize[1] ;
              console.log("Clicked image: " + x + "," + y)
              if(x && y)
              request("./deforum/live-edit/look-at", { "x": x, "y": y }, function (res) {
                console.log("Set look at point",res)
              }, function(err){
                console.log("Failed to set look at point",err)
              });
              /*request("./internal/progress", { "id_task": id_task, "id_live_preview": id_live_preview }, function (res) {

              }, function () { })*/
            }
          }
        }
      }
    });
  });
  observer.observe(targetNode, { childList: true, subtree: true });

});

function getContainedSize(img) {
  var ratio = img.naturalWidth/img.naturalHeight
  var width = img.height*ratio
  var height = img.height
  if (width > img.width) {
    width = img.width
    height = img.width/ratio
  }
  return [width, height]
}