from ...domain.entities.speaker import SpeakerEmbedding
from ...domain.interfaces.voice_cloner import VoiceCloner


class CloneVoiceUseCase:
    """Create speaker embeddings from audio samples."""
    
    def __init__(self, voice_cloner: VoiceCloner):
        self.voice_cloner = voice_cloner
    
    async def execute(self, audio_path: str, speaker_id: str) -> SpeakerEmbedding:
        """Clone voice and save speaker embedding."""
        embedding = await self.voice_cloner.clone_from_sample(audio_path)
        
        # Set speaker ID
        embedding.speaker_id = speaker_id
        
        # Save for future use
        self.voice_cloner.save_speaker(embedding, speaker_id)
        
        return embedding
