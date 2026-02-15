from abc import ABC, abstractmethod


# Target interface
class MediaPlayer(ABC):
    @abstractmethod
    def play(self, filename: str):
        pass


# Adaptee
class MP3Player:
    def play_mp3(self, filename: str):
        print(f"🎵 Playing MP3: {filename}")


# Class Adapter using multiple inheritance
class MP3ClassAdapter(MediaPlayer, MP3Player):
    """
    Uses multiple inheritance.
    Inherits from both target interface and adaptee.
    """

    def play(self, filename: str):
        # Call inherited method from MP3Player
        self.play_mp3(filename)


if __name__ == '__main__':
    adapter = MP3ClassAdapter()
    adapter.play("song.mp3")