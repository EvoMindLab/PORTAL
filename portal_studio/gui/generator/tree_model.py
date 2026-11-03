"""Tree-selection helpers for the Generator tab."""

from __future__ import annotations

from typing import Any

from PySide6 import QtCore
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem

from portal_studio.experiment.session import ExpBlock, ExpComponent

_ROLE = QtCore.Qt.ItemDataRole.UserRole


def item_role_data(item: QTreeWidgetItem | None) -> Any | None:
    return None if item is None else item.data(0, _ROLE)


def selected_role_data(tree: QTreeWidget) -> Any | None:
    items = tree.selectedItems()
    return item_role_data(items[0]) if items else None


def active_block_from_tree(tree: QTreeWidget) -> ExpBlock | None:
    items = tree.selectedItems()
    if not items:
        return None
    item = items[0]
    while item.parent() is not None:
        item = item.parent()
    data = item_role_data(item)
    return data if isinstance(data, ExpBlock) else None


def find_block_item(tree: QTreeWidget, block: ExpBlock) -> QTreeWidgetItem | None:
    for i in range(tree.topLevelItemCount()):
        node = tree.topLevelItem(i)
        if item_role_data(node) is block:
            return node
    return None


def find_component_item(
    tree: QTreeWidget,
    block: ExpBlock,
    component: ExpComponent,
) -> QTreeWidgetItem | None:
    node = find_block_item(tree, block)
    if node is None:
        return None
    for j in range(node.childCount()):
        child = node.child(j)
        if item_role_data(child) is component:
            return child
    return None
