/**
 * This is the BST ADT. It should contain methods that allow it to insert new
 * nodes, delete nodes, search, etc. You'll be adding code to this class for
 * this lab.
 * 
 * 
 * @author Xavier
 * @version 03.02.17
 */
public class BinarySearchTree<E extends Comparable<E>> {
	private BSTNode<E> root;
	private int size = 0;

	/**
	 * Insterts the given data into the tree
	 * 
	 * @param whatever
	 *            you want to put in the tree
	 * 
	 **/
	public void insert(E data) {
		BSTNode toInsert = new BSTNode(data);

		if (root == null) {
			root = toInsert;
		} else {
			BSTNode<E> n = root;
			BSTNode<E> prev = null;
			while (n != null) {
				if (toInsert.key.compareTo(n.key) > 0) {
					prev = n;
					n = n.rLink;
				} else {
					prev = n;
					n = n.lLink;
				}
			}

			if (toInsert.key.compareTo(prev.key) > 0) {
				prev.rLink = toInsert;
			} else {
				prev.lLink = toInsert;
			}
		}
		size++;
	}

	/**
	 * Searches for a node with specific data and returns it. Returns null if
	 * its not found.
	 * 
	 * @param node
	 *            The node to start at (usually root)
	 * @param value
	 *            The data that you're looking for
	 * @return The node, or null
	 */
	private BSTNode<E> search(BSTNode<E> node, E value) {

		if (node == null) {
			return null;
		}

		if (node.key.compareTo(value) == 0) {
			return node;
		} else {
			BSTNode left = search(node.lLink, value);
			BSTNode right = search(node.rLink, value);

			if (left != null) {
				return left;
			} else {
				return right;
			}
		}
	}

	/**
	 * Searches for some data and returns its node, or null if the node is not
	 * found
	 * 
	 * @param data
	 *            Whatever you want to find
	 * @return The node, or null
	 */
	public E search(E data) {
		return search(root, data).key;
	}

	/**
	 * Counts how many nodes there are in the tree
	 * 
	 * @return Number of nodes in the tree
	 */
	public int getSize() {
		return size;
	}

	/**
	 * A general toString method
	 * 
	 * @return a string off the entire tree including parens to denote children
	 */
	public String toString() {
		return stringNodes(root);
	}

	/**
	 * Returns a string off the entire tree including parens to denote children
	 * 
	 * @param N
	 *            the node to start at (usually root)
	 * @return The string version of all the nodes.
	 */
	private String stringNodes(BSTNode<E> N) {
		String toReturn = "";
		if (N != null) {
			toReturn += "(";
			toReturn += stringNodes(N.lLink);
			toReturn += "  " + N.key + " ";
			toReturn += stringNodes(N.rLink);
			toReturn += ")";

		}
		return toReturn;
	}
	
	
}