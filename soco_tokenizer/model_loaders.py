from soco_encoders.Tokenizer import Tokenizer


class EncoderLoader(object):
    @staticmethod
    def load_tokenizer(model_id, region='us'):
        return Tokenizer(model_id, region)

