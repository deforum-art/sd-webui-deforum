
onUiLoaded(function () {
  const galleryNode = gradioApp().querySelector("#deforum_gallery_container");
  const observer = new MutationObserver(function (mutations) {
    mutations.forEach(function (mutation) {
      if (mutation.type === 'childList') {
        const addedNodes = mutation.addedNodes;
        for (let i = 0; i < addedNodes.length; i++) {
          let node = addedNodes[i];
          // There can be multiple livePreview nodes, so we apply the click listener to all of them
          if (node.classList && node.classList.contains('livePreview')) {
            node.onclick = function (event) {
              const imageNode = event.target
              const containedSize = getContainedSize(imageNode);
              const leftOffset = (imageNode.width - containedSize[0]) / 2;
              const topOffset = (imageNode.height - containedSize[1]) / 2;
              var x01 = (event.offsetX - leftOffset) / containedSize[0] ;
              var y01 = (event.offsetY - topOffset) / containedSize[1] ;
              if(x01 && y01)
              request("./deforum/live-edit/look-at", { "x": x01, "y": y01 }, function (res) {
              }, function(err){
                console.error("Failed to set look at point",err)
              });
            }
          }
        }
      }
    });
  });
  observer.observe(galleryNode, { childList: true, subtree: true });

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