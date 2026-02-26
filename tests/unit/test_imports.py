def test_package_imports():
    import skedl

    assert hasattr(skedl, "__version__")
