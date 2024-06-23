from stock_tools.utils.config import Config


def test_config():
    cfg = Config().get_config()
    assert isinstance(cfg, dict)
