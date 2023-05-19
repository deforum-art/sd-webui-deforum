##
# From /scripts directory, run like: python -m unittest deforum_helpers.parseq_adapter_test
##

import unittest
from .parseq_adapter import ParseqAnimKeys 
from .animation_key_frames import DeformAnimKeys
from unittest.mock import patch
from unittest.mock import MagicMock, PropertyMock

from types import SimpleNamespace

class TestParseqAnimKeys(unittest.TestCase):

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    def test_withprompt(self, mock_deformanimkeys):
        parseq_args = SimpleNamespace(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "deforum_prompt": "blah"
                    },
                    {
                        "frame": 1,
                        "deforum_prompt": "blah"
                    }
                ]
            }
            """)
        anim_args = SimpleNamespace(max_frames=2)
        video_args = SimpleNamespace(fps=30)
        parseq_anim_keys = ParseqAnimKeys(parseq_args, anim_args, video_args)
        self.assertTrue(parseq_anim_keys.manages_prompts())


    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    def test_withoutprompt(self, mock_deformanimkeys):
        parseq_args = SimpleNamespace(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        anim_args = SimpleNamespace(max_frames=2)
        video_args = SimpleNamespace(fps=30)
        parseq_anim_keys = ParseqAnimKeys(parseq_args, anim_args, video_args)
        self.assertFalse(parseq_anim_keys.manages_prompts())

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    def test_usedelta(self, mock_deformanimkeys):
        parseq_args = SimpleNamespace(parseq_use_deltas=True, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "angle": 90,
                        "angle_delta": 90
                    },
                    {
                        "frame": 1,
                        "angle": 180,
                        "angle_delta": 90
                    }
                ]
            }
            """)
        anim_args = SimpleNamespace(max_frames=2)
        video_args = SimpleNamespace(fps=30)
        parseq_anim_keys = ParseqAnimKeys(parseq_args, anim_args, video_args)
        self.assertEqual(parseq_anim_keys.angle_series[1], 90)

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    def test_usenondelta(self, mock_deformanimkeys):
        parseq_args = SimpleNamespace(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0,
                        "angle": 90,
                        "angle_delta": 90
                    },
                    {
                        "frame": 1,
                        "angle": 180,
                        "angle_delta": 90
                    }
                ]
            }
            """)
        anim_args = SimpleNamespace(max_frames=2)
        video_args = SimpleNamespace(fps=30)
        parseq_anim_keys = ParseqAnimKeys(parseq_args, anim_args, video_args)
        self.assertEqual(parseq_anim_keys.angle_series[1], 180)

    @patch('deforum_helpers.parseq_adapter.DeformAnimKeys')
    def test_fallbackonundefined(self, mock_deformanimkeys):
        parseq_args = SimpleNamespace(parseq_use_deltas=False, parseq_manifest=""" 
            {                
                "options": {
                    "output_fps": 30
                },
                "rendered_frames": [
                    {
                        "frame": 0
                    },
                    {
                        "frame": 1
                    }
                ]
            }
            """)
        
        anim_args = SimpleNamespace(max_frames=1)
        video_args = SimpleNamespace(fps=20)
        parseq_anim_keys = ParseqAnimKeys(parseq_args, anim_args, video_args)
        #TODO - this is a hacky check to make sure we're falling back to the mock.
        #There must be a better way to inject an expected value via patch and check for that...
        self.assertRegex(str(parseq_anim_keys.angle_series[0]), r'MagicMock')
        
if __name__ == '__main__':
    unittest.main()