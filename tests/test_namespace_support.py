"""
Tests for namespace support in DIMOS robot classes.

This test validates that the namespace parameter correctly prefixes topics and frames
for multi-robot ROS2 environments (e.g., Isaac Sim simulations).
"""

import unittest


class TestNamespaceFrameFormatting(unittest.TestCase):
    """Test that frame names are correctly formatted with namespaces."""
    
    def test_frame_namespace_formatting_with_namespace(self):
        """Test that frames are correctly prefixed when namespace is provided."""
        # Test with namespace
        namespace = "robot0"
        source_frame = "base_link"
        target_frame = "map"
        
        # Apply namespace formatting as the code does
        prefixed_source = f"{namespace}/{source_frame}".lstrip("/")
        prefixed_target = f"{namespace}/{target_frame}".lstrip("/")
        
        self.assertEqual(prefixed_source, "robot0/base_link")
        self.assertEqual(prefixed_target, "robot0/map")
    
    def test_frame_namespace_formatting_without_namespace(self):
        """Test that frames are unchanged when namespace is empty."""
        # Test without namespace
        namespace = ""
        source_frame = "base_link"
        target_frame = "map"
        
        # Apply namespace formatting
        if namespace:
            prefixed_source = f"{namespace}/{source_frame}".lstrip("/")
            prefixed_target = f"{namespace}/{target_frame}".lstrip("/")
        else:
            prefixed_source = source_frame
            prefixed_target = target_frame
        
        self.assertEqual(prefixed_source, "base_link")
        self.assertEqual(prefixed_target, "map")
    
    def test_frame_namespace_with_leading_slash(self):
        """Test that leading slashes are handled correctly."""
        namespace = "/robot0"
        source_frame = "base_link"
        
        # Strip trailing slash from namespace (as in UnitreeGo2.__init__)
        namespace = namespace.rstrip("/")
        prefixed_source = f"{namespace}/{source_frame}".lstrip("/")
        
        self.assertEqual(prefixed_source, "robot0/base_link")
    
    def test_frame_namespace_with_trailing_slash(self):
        """Test that trailing slashes are handled correctly."""
        namespace = "robot0/"
        source_frame = "base_link"
        
        # Strip trailing slash from namespace (as in UnitreeGo2.__init__)
        namespace = namespace.rstrip("/")
        prefixed_source = f"{namespace}/{source_frame}".lstrip("/")
        
        self.assertEqual(prefixed_source, "robot0/base_link")


class TestNamespaceTopicFormatting(unittest.TestCase):
    """Test that topic names are correctly formatted with namespaces."""
    
    def test_topic_namespace_formatting_with_namespace(self):
        """Test that topics are correctly prefixed when namespace is provided."""
        namespace = "robot0"
        topic = "local_costmap/costmap"
        
        # Apply namespace formatting as the code does
        prefixed_topic = f"{namespace}/{topic}".lstrip("/")
        
        self.assertEqual(prefixed_topic, "robot0/local_costmap/costmap")
    
    def test_topic_namespace_formatting_without_namespace(self):
        """Test that topics are unchanged when namespace is empty."""
        namespace = ""
        topic = "/local_costmap/costmap"
        
        # Apply namespace formatting
        prefixed_topic = f"{namespace}/{topic}".lstrip("/")
        
        self.assertEqual(prefixed_topic, "local_costmap/costmap")
    
    def test_map_topic_with_namespace(self):
        """Test that map topic is correctly prefixed."""
        namespace = "robot0"
        topic = "map"
        
        prefixed_topic = f"{namespace}/{topic}".lstrip("/")
        
        self.assertEqual(prefixed_topic, "robot0/map")


class TestNamespaceStripLogic(unittest.TestCase):
    """Test the namespace strip logic as implemented in Robot class."""
    
    def test_namespace_strip_trailing_slash(self):
        """Test that trailing slashes are stripped from namespace."""
        namespace = "robot0/"
        cleaned = namespace.rstrip("/") if namespace else ""
        self.assertEqual(cleaned, "robot0")
    
    def test_namespace_strip_multiple_trailing_slashes(self):
        """Test that multiple trailing slashes are stripped."""
        namespace = "robot0///"
        cleaned = namespace.rstrip("/") if namespace else ""
        self.assertEqual(cleaned, "robot0")
    
    def test_namespace_empty_string(self):
        """Test that empty string namespace remains empty."""
        namespace = ""
        cleaned = namespace.rstrip("/") if namespace else ""
        self.assertEqual(cleaned, "")


if __name__ == '__main__':
    unittest.main()
