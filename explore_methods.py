from urielplus import urielplus

def explore_urielplus_methods():
    """Explore all available methods in URIELPlus"""

    u = urielplus.URIELPlus()

    print("=== All URIELPlus methods ===")
    all_methods = [method for method in dir(u) if not method.startswith('_')]

    # Group methods by type
    get_methods = [m for m in all_methods if m.startswith('get_')]
    new_methods = [m for m in all_methods if m.startswith('new_')]
    vector_methods = [m for m in all_methods if 'vector' in m.lower()]
    feature_methods = [m for m in all_methods if 'feature' in m.lower()]

    print(f"\nMethods starting with 'get_' ({len(get_methods)}):")
    for method in sorted(get_methods):
        print(f"  - {method}")

    print(f"\nMethods starting with 'new_' ({len(new_methods)}):")
    for method in sorted(new_methods):
        print(f"  - {method}")

    print(f"\nMethods containing 'vector' ({len(vector_methods)}):")
    for method in sorted(vector_methods):
        print(f"  - {method}")

    print(f"\nMethods containing 'feature' ({len(feature_methods)}):")
    for method in sorted(feature_methods):
        print(f"  - {method}")

    print(f"\nAll other methods ({len(all_methods) - len(get_methods) - len(new_methods)}):")
    other_methods = [m for m in all_methods if not m.startswith('get_') and not m.startswith('new_')]
    for method in sorted(other_methods):
        print(f"  - {method}")

if __name__ == "__main__":
    explore_urielplus_methods()