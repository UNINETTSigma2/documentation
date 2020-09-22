import mistune
import collections

markdown = mistune.create_markdown(renderer=mistune.AstRenderer())


def get_tree(file_name):
    tree = collections.defaultdict(list)
    with open(file_name, "r") as f:
        for x in markdown(f.read()):
            if x["level"] == 2:
                section_title = x["children"][0]["text"]
            if x["level"] == 1:
                for child in x["children"]:
                    _page_title = child["children"][0]["children"][0]["children"][0][
                        "text"
                    ]
                    md_file_name = child["children"][0]["children"][0]["link"]
                    tree[section_title].append(md_file_name)
    return tree


def print_header():
    print(
        """
===================================
The Norwegian Academic HPC Services
==================================="""
    )


def print_section(section, file_names):
    print("\n\n.. toctree::")
    print("   :maxdepth: 1")
    print(f"   :caption: {section}\n")
    for file_name in file_names:
        print(f"   {file_name}")


tree = get_tree("SUMMARY.md")

print_header()
for section in tree:
    print_section(section, tree[section])
