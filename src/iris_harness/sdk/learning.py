"""Reading and writing what the harness has learned.

`LessonCapture` is prior attempts at a similar task plus where a run's outcome
is recorded -- an agent plugin primes itself from it and writes back through it.
`SurfaceFeedbackStore` is how the user has steered a surface, which a plugin
reads to stop offering what was rejected.

`HarnessServices.lessons` hands the same store to a plugin that has one; this
module is for the plugin that must construct its own.
"""

from __future__ import annotations

from iris_harness.services.learning.lesson_capture import LessonCapture
from iris_harness.services.learning.suppression import SurfaceFeedbackStore

__all__ = ["LessonCapture", "SurfaceFeedbackStore"]
