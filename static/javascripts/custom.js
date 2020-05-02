//Gioi han checkbox
var limit = 2;
$('input.single-checkbox').on('change', function(evt) {
    if ($(this).siblings(':checked').length >= limit) {
        this.checked = false;
    }
});


//Phat ra tieng khi nhan submit
function play() {
    var audio = document.getElementById("audio");
    audio.play();
  }