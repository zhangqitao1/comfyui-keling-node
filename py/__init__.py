from .nodes import Text2VideoNode, ImageGeneratorNode, TextToAudioNode, Video2AudioNode, ImageExpanderNode, \
    Image2VideoNode, KLingAIAPIClient, PreviewVideo, PreviewAudio, KolorsVirtualTryOnNode, VideoExtendNode, LipSyncNode, \
    LipSyncTextInputNode, LipSyncAudioInputNode, EffectNode, MultiImagesToVideoNode, Text2VideoO1Node, Image2VideoO1Node, \
    UploadVideoNode

NODE_CLASS_MAPPINGS = {
    'Client': KLingAIAPIClient,
    'Image Generator': ImageGeneratorNode,
    'Image Expander': ImageExpanderNode,
    "Text2Audio": TextToAudioNode,
    "Video2Audio": Video2AudioNode,
    'Text2Video': Text2VideoNode,
    'Image2Video': Image2VideoNode,
    'Text2VideoO1': Text2VideoO1Node,
    'Image2VideoO1': Image2VideoO1Node,
    'MultiImages2Video': MultiImagesToVideoNode,
    'Virtual Try On': KolorsVirtualTryOnNode,
    'KLingAI Preview Video': PreviewVideo,
    'KLingAI Preview Audio': PreviewAudio,
    'Video Extender': VideoExtendNode,
    'Lip Sync': LipSyncNode,
    'Lip Sync Text Input': LipSyncTextInputNode,
    'Lip Sync Audio Input': LipSyncAudioInputNode,
    'Effects': EffectNode,
    'Upload Video': UploadVideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'Client': 'KLingAI Client',
    'Image Generator': 'KLingAI Image Generator',
    'Image Expander': 'KLingAI Image Expander',
    "Text2Audio": 'KLingAI Text2Audio',
    "Video2Audio": 'KLingAI Video2Audio',
    'Text2Video': 'KLingAI Text2Video',
    'Image2Video': 'KLingAI Image2Video',
    'Text2VideoO1': 'KLingAI Text2Video O1',
    'Image2VideoO1': 'KLingAI Image2Video O1',
    'MultiImages2Video': 'KLingAI MultiImages2Video',
    'Virtual Try On': 'KLingAI Virtual Try On',
    'KLingAI Preview Video': 'KLingAI Preview Video',
    'KLingAI Preview Audio': 'KLingAI Preview Audio',
    'Video Extender': 'KLingAI Video Extender',
    'Lip Sync': 'KLingAI Lip Sync',
    'Lip Sync Text Input': 'KLingAI Lip Sync Text Input',
    'Lip Sync Audio Input': 'KLingAI Lip Sync Audio Input',
    'Effects': 'KLingAI Effects',
    'Upload Video': 'KLingAI Upload Video'
}
