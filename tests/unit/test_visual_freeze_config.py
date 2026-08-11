from molt.models.base import _automodel_freeze_config


def test_visual_freeze_config_is_disabled_by_default():
    assert _automodel_freeze_config(False) is None


def test_visual_freeze_config_freezes_only_the_vision_tower():
    assert _automodel_freeze_config(True) == {
        "freeze_vision_tower": True,
        "freeze_audio_tower": False,
        "freeze_language_model": False,
        "freeze_video_embedder": False,
    }
