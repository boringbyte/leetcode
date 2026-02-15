from abc import ABC, abstractmethod


# ============ TARGET INTERFACE ============
# This is what the client expects

class MediaPlayer(ABC):
    """Target interface - what client code expects"""

    @abstractmethod
    def play(self, filename: str):
        pass


# ============ ADAPTEE ============
# These are the incompatible classes we want to use

class MP3Player:
    """Existing class with incompatible interface"""

    def play_mp3(self, filename: str):
        print(f"🎵 Playing MP3 file: {filename}")


class MP4Player:
    """Another existing class with different interface"""

    def play_mp4(self, filename: str):
        print(f"🎬 Playing MP4 file: {filename}")


class VLCPlayer:
    """Yet another class with its own interface"""

    def play_vlc(self, filename: str):
        print(f"🎥 Playing VLC file: {filename}")


# ============ ADAPTERS ============
# These adapt the incompatible interfaces to our target interface

class MP3Adapter(MediaPlayer):
    """Adapter for MP3Player"""

    def __init__(self):
        self.mp3_player = MP3Player()  # Composition

    def play(self, filename: str):
        # Adapt the interface
        self.mp3_player.play_mp3(filename)


class MP4Adapter(MediaPlayer):
    """Adapter for MP4Player"""

    def __init__(self):
        self.mp4_player = MP4Player()  # Composition

    def play(self, filename: str):
        self.mp4_player.play_mp4(filename)


class VLCAdapter(MediaPlayer):
    """Adapter for VLCPlayer"""

    def __init__(self):
        self.vlc_player = VLCPlayer()  # Composition

    def play(self, filename: str):
        self.vlc_player.play_vlc(filename)


# ============ CLIENT CODE ============

class AudioPlayer:
    """
    Client code that works with MediaPlayer interface.
    It doesn't need to know about MP3Player, MP4Player, etc.
    """

    def __init__(self):
        self.players = {
            'mp3': MP3Adapter(),
            'mp4': MP4Adapter(),
            'vlc': VLCAdapter()
        }

    def play(self, file_type: str, filename: str):
        player = self.players.get(file_type.lower())

        if player:
            player.play(filename)
        else:
            print(f"❌ Unsupported file type: {file_type}")


if __name__ == '__main__':
    # Usage
    audio_player = AudioPlayer()

    audio_player.play('mp3', 'song.mp3')
    audio_player.play('mp4', 'video.mp4')
    audio_player.play('vlc', 'movie.vlc')
    audio_player.play('avi', 'video.avi')  # Unsupported
