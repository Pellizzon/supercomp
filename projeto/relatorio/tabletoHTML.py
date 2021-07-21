import markdown

table_md = """| Item 37 | Item 20 | Item 11 | Valor 0 | Valor 1 |
| :-----: | :-----: | :-----: | :-----: | :-----: |
|    0    |    0    |    0    |   68    |    0    |
|    0    |    0    |    1    |   57    |   11    |
|    0    |    1    |    0    |   48    |   20    |
|    0    |    1    |    1    |   37    |   31    |
|    1    |    0    |    0    |   11    |   57    |
|    1    |    0    |    1    |   20    |   48    |
|    1    |    1    |    0    |   11    |   57    |
|    1    |    1    |    1    |    0    |   68    |"""

table_html = markdown.markdown(table_md, extensions=["markdown.extensions.tables"])

print(table_html)