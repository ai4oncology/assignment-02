from assignment import class_names


def test_class_names():
    result = class_names()
    assert result == ["glioma", "meningioma", "no_tumor", "pituitary"]
