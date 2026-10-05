from orders import describe_order


def test_describe_pending():
    assert describe_order("pending") == "Pending"


def test_describe_shipped():
    assert describe_order("shipped") == "Shipped"


def test_describe_delivered():
    assert describe_order("delivered") == "Delivered"
