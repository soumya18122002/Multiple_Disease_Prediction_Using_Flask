// window.addEventListener("load",()=>{
//     const loader = document.querySelector(".loader");

//     loader.classList.add("loader-hidden");
//     loader.addEventListener("transitioned", ()=>{
//         document.body.removeChild("loader");
//     })
// })
// setTimeout(function(){
//     $('loader_bg').fadeToggle();
// },1500);
// $(window).on('load', function () {
//     $('#loading').hide();
//   }); 

$(document).ajaxStart(function () {
  $("#loader").show();
});

$(document).ajaxStop(function () {
  $("#loader").hide();
});

$.ajax({
  url: '/api/data',
  type: 'GET',
  beforeSend: function () {
    // Show loader here
    $('#loader').show();
  },
  success: function (data) {
    // Handle data here
  },
  complete: function () {
    // Hide loader here
    $('#loader').hide();
  }
});
var request = $.ajax({
  url: '/api/data',
  type: 'GET',
  beforeSend: function () {
    // Show loader here
    $('#loader').show();
  },
  success: function (data) {
    // Handle data here
  },
  complete: function () {
    // Hide loader here
    $('#loader').hide();
  }
});

// To stop the request:
request.abort();