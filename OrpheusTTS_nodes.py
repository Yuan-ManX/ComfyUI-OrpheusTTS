import wave
import time
import os
import re
import asyncio
import logging

from orpheus_tts import OrpheusModel


class SingleTextGeneration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": ""}),
                "voice": ("STRING", {"default": "tara", "choices": ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 1.1, "max": 2.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 16384, "step": 128}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    FUNCTION = "generate_speech"
    CATEGORY = "OrpheusTTS"
    
    def __init__(self):
        self.model = None
        self.model_sample_rate = 24000
        self.model_path = None
        self.model_name = self.model_path if self.model_path else "canopylabs/orpheus-tts-0.1-finetune-prod"
    
    def load_model(self):
        if self.model is None:
            logging.info(f"Loading model from: {self.model_name}")
            self.model = OrpheusModel(model_name=self.model_name)
    
    def generate_speech(self, prompt, voice, temperature, top_p, repetition_penalty, max_tokens):
        self.load_model()
        
        start_time = time.monotonic()
        
        syn_tokens = self.model.generate_speech(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens
        )
        
        output_filename = f"output_{int(time.time())}.wav"
        
        with wave.open(output_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.model_sample_rate)
            
            total_frames = 0
            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
        
        processing_time = time.monotonic() - start_time
        result_message = f"Generated {duration:.2f} seconds of audio in {processing_time:.2f} seconds"
        logging.info(result_message)

        return (output_filename, result_message)


class LongFormContent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "long_text": ("STRING", {"default": ""}),
                "voice": ("STRING", {"default": "tara", "choices": ["tara", "jess", "leo", "leah", "dan", "mia", "zac", "zoe"]}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.1, "max": 2.0, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "repetition_penalty": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "max_tokens": ("INT", {"default": 4096, "min": 128, "max": 16384, "step": 128}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "STRING")
    FUNCTION = "generate_long_form_speech"
    CATEGORY = "OrpheusTTS"
    
    def __init__(self):
        self.model = None
        self.model_sample_rate = 24000
        self.model_path = None
        self.model_name = self.model_path if self.model_path else "canopylabs/orpheus-tts-0.1-finetune-prod"
    
    def load_model(self):
        if self.model is None:
            logging.info(f"Loading model from: {self.model_name}")
            self.model = OrpheusModel(model_name=self.model_name)
    
    def chunk_text(self, text, max_chunk_size=300):
        text = re.sub(r"\s+", " ", text)
        delimiter_pattern = r'(?<=[.!?])\s+'
        segments = re.split(delimiter_pattern, text)
        sentences = []
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            if not segment[-1] in ['.', '!', '?']:
                segment += '.'
            sentences.append(segment)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        if current_chunk:
            chunks.append(current_chunk)
        logging.info(f"Text chunked into {len(chunks)} segments")
        return chunks
    
    async def process_chunk(self, chunk, voice, temperature, top_p, repetition_penalty, max_tokens, temp_dir, current_idx):
        loop = asyncio.get_event_loop()
        def generate_for_chunk():
            return self.model.generate_speech(
                prompt=chunk,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_tokens=max_tokens
            )
        syn_tokens = await loop.run_in_executor(None, generate_for_chunk)
        chunk_filename = os.path.join(temp_dir, f"chunk_{current_idx}.wav")
        with wave.open(chunk_filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.model_sample_rate)
            chunk_frames = 0
            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                chunk_frames += frame_count
                wf.writeframes(audio_chunk)
            chunk_duration = chunk_frames / wf.getframerate()
        return chunk_filename, chunk_duration
    
    async def generate_long_form_speech_async(self, long_text, voice, temperature, top_p, repetition_penalty, batch_size, max_tokens):
        start_time = time.monotonic()
        chunks = self.chunk_text(long_text)
        temp_dir = f"longform_{int(time.time())}"
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Created temp directory: {temp_dir}")
        semaphore = asyncio.Semaphore(batch_size)
        total_chunks = len(chunks)
        all_audio_files = []
        total_duration = 0
        async def process_chunk_with_semaphore(chunk, idx):
            async with semaphore:
                try:
                    filename, duration = await self.process_chunk(
                        chunk, voice, temperature, top_p, repetition_penalty,
                        max_tokens, temp_dir, idx
                    )
                    return filename, duration
                except Exception as e:
                    logging.error(f"Error processing chunk {idx}: {str(e)}")
                    raise
        tasks = [process_chunk_with_semaphore(chunk, idx) for idx, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        for filename, duration in results:
            all_audio_files.append(filename)
            total_duration += duration
        combined_filename = f"longform_output_{int(time.time())}.wav"
        logging.info(f"Combining {len(all_audio_files)} audio chunks into {combined_filename}")
        data = []
        for file in sorted(all_audio_files, key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0])):
            with wave.open(file, 'rb') as w:
                data.append([w.getparams(), w.readframes(w.getnframes())])
        with wave.open(combined_filename, 'wb') as output:
            if data:
                output.setparams(data[0][0])
                for i in range(len(data)):
                    output.writeframes(data[i][1])
        for file in all_audio_files:
            try:
                os.remove(file)
            except Exception as e:
                logging.warning(f"Failed to delete temp file {file}: {e}")
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            logging.warning(f"Failed to delete temp directory {temp_dir}: {e}")
        processing_time = time.monotonic() - start_time
        result_message = f"Generated {total_duration:.2f} seconds of audio from {total_chunks} chunks in {processing_time:.2f} seconds"
        logging.info(result_message)
        return combined_filename, result_message
    
    def generate_long_form_speech(self, long_text, voice, temperature, top_p, repetition_penalty, batch_size, max_tokens):
        self.load_model()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            combined_filename, result_message = loop.run_until_complete(
                self.generate_long_form_speech_async(
                    long_text, voice, temperature, top_p,
                    repetition_penalty, batch_size, max_tokens
                )
            )
        finally:
            loop.close()
        return combined_filename, result_message

