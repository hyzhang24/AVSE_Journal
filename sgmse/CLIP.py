from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection

class Clip4Video(nn.Module):
    def __init__(self, model, embedding_dim=1024, pretrained=True, pe=False):
        super(Clip4Video, self).__init__()
        self.pretrained = pretrained
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(model)
        self.clip_text = CLIPTextModelWithProjection.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        input_dim = 512 if "clip-vit-base" in model else 768
        self.linear_layer = nn.Linear(input_dim, embedding_dim)
        self.linear_layer1 = nn.Linear(309, embedding_dim)
        self.pe = sinusoidal_positional_embedding(30, input_dim) if pe else None
        print("*****PE*****") if pe else print("*****W/O PE*****")

    def forward(self, text=None, image=None, video=None):
        assert text is not None or image is not None or video is not None, "At least one of text, image or video should be provided"
        if text is not None and video is None:
            inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=77).to(self.clip_text.device)
            out = self.clip_text(**inputs)
            out = out.text_embeds.repeat(20, 1)
        elif video is not None and text is None:
            out = self.clip_vision(video.to(self.clip_vision.device))          # input video x: t * 3 * w * h
            out = out.image_embeds        # t * 512
            if self.pe is not None:
                out = out + self.pe[:out.shape[0], :].to(self.clip_vision.device)
            # out['last_hidden_state'].shape # t * 50 * 768
            # out['image_embeds'].shape      # t * 512
        elif text is not None and video is not None:
            text_inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=77).to(self.clip_text.device)
            video_out = self.clip_vision(video.to(self.clip_vision.device))
            video_out = video_out.image_embeds
            if self.pe is not None:
                video_out = video_out + self.pe[:video_out.shape[0], :].to(self.clip_vision.device)

            text_out = self.clip_text(**text_inputs).text_embeds
            # text_out = text_out.repeat(video_out.shape[0], 1)
            # concat
            # out = torch.cat([text_out, video_out], dim=0)
            video_out = self.linear_layer(video_out)
            text_out = self.linear_layer(text_out)

            return video_out, text_out

        # print(out.shape)    # torch.Size([21, 768])
        out = self.linear_layer(out)     # t * 1024
        return out