class audio_player {
    static make_audio(filename) {
    this.sound = new Howl({
    src: [filename]
    });
    }
   static play_audio() {
    this.sound.play();
    }
   static stop_audio() {
    if (this.sound != null) {
        this.sound.stop();
        this.sound.unload();
        this.sound = null;   }
}
}