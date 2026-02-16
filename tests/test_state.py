"""Tests for the scene state calculator (ground truth engine)."""

from generator.state import ModifiableParams, calculate_scene


class TestCalculateScene:
    """Tests for calculate_scene()."""

    def test_deterministic_same_seed(self):
        """Same params + seed should produce identical scenes."""
        params = ModifiableParams(item_count=5, spacing_px=40)
        scene_a = calculate_scene(params, seed=42)
        scene_b = calculate_scene(params, seed=42)

        assert len(scene_a.items) == len(scene_b.items)
        for a, b in zip(scene_a.items, scene_b.items):
            assert a.x == b.x
            assert a.y == b.y

    def test_ground_truth_counting(self):
        """Ground truth should match the item_count parameter."""
        for count in [1, 3, 7, 15]:
            params = ModifiableParams(item_count=count)
            scene = calculate_scene(params, seed=0)
            assert scene.ground_truth_answer == str(count)

    def test_different_seeds_different_positions(self):
        """Different seeds should produce different item positions."""
        params = ModifiableParams(item_count=5, spacing_px=30)
        scene_a = calculate_scene(params, seed=0)
        scene_b = calculate_scene(params, seed=1)

        # At least one item should be in a different position
        positions_a = [(i.x, i.y) for i in scene_a.items]
        positions_b = [(i.x, i.y) for i in scene_b.items]
        assert positions_a != positions_b

    def test_correct_item_count(self):
        """Scene should contain exactly item_count items."""
        for count in [1, 5, 10]:
            params = ModifiableParams(item_count=count, spacing_px=20)
            scene = calculate_scene(params, seed=0)
            assert len(scene.items) == count

    def test_distractor_generation(self):
        """Distractors should appear when enabled."""
        params = ModifiableParams(
            item_count=3,
            distractor_presence=True,
            distractor_count=4,
        )
        scene = calculate_scene(params, seed=0)
        assert len(scene.items) == 3
        assert len(scene.distractors) == 4

    def test_no_distractors_by_default(self):
        """No distractors when distractor_presence is False."""
        params = ModifiableParams(item_count=3)
        scene = calculate_scene(params, seed=0)
        assert len(scene.distractors) == 0

    def test_question_includes_shape(self):
        """Question should reference the correct shape."""
        for shape in ["circle", "square", "triangle"]:
            params = ModifiableParams(shape=shape)
            scene = calculate_scene(params, seed=0)
            assert shape in scene.question

    def test_modifiable_params_roundtrip(self):
        """ModifiableParams should survive to_dict/from_dict roundtrip."""
        orig = ModifiableParams(
            item_count=7,
            spacing_px=50,
            background_color="#000000",
            distractor_presence=True,
        )
        d = orig.to_dict()
        restored = ModifiableParams.from_dict(d)
        assert restored.item_count == orig.item_count
        assert restored.spacing_px == orig.spacing_px
        assert restored.background_color == orig.background_color
        assert restored.distractor_presence == orig.distractor_presence
