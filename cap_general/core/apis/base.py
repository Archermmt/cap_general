"""Base class for CAP APIs with documentation extraction."""

import inspect
from typing import List, Dict, Any


class CapApiBase:
    """Base class for CAP APIs that provides automatic documentation extraction.

    Subclasses should implement robot/environment control methods that will be
    exposed to the policy model for code generation.
    """

    def combined_doc(self) -> str:
        """Extract and combine documentation from all public methods.

        Returns:
            A formatted string containing method signatures and docstrings.
        """
        docs = []

        # Get all public methods (not starting with _)
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            # Get signature
            sig = inspect.signature(method)
            sig_str = f"{name}{sig}"

            # Get docstring
            docstring = inspect.getdoc(method)

            # Combine
            doc_section = f"def {sig_str}:"
            if docstring:
                doc_section += f'\n    """{docstring}"""'

            docs.append(doc_section)

        return "\n\n".join(docs)

    def api_spec(self) -> Dict[str, Any]:
        """Get API specification as a dictionary.

        Returns:
            Dictionary mapping method names to their specifications.
        """
        spec = {}

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            sig = inspect.signature(method)
            docstring = inspect.getdoc(method)

            spec[name] = {
                "signature": str(sig),
                "docstring": docstring,
                "parameters": {
                    param_name: {
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else "Any"
                        ),
                        "default": (
                            str(param.default)
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                    }
                    for param_name, param in sig.parameters.items()
                },
                "return_annotation": (
                    str(sig.return_annotation)
                    if sig.return_annotation != inspect.Signature.empty
                    else "Any"
                ),
            }

        return spec
