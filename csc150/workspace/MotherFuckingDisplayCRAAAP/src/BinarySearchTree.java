/**
 * This is the BST ADT. It should contain methods that allow it to insert new
 * nodes, delete nodes, search, etc. You'll be adding code to this class for
 * this lab.
 * 
 * @author (your name)
 * @version (a version number or a date)
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
	public BSTNode search(E data) {
		return search(root, data);
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

	/**
	 * Gets the height of the whole tree
	 * @param N the node to start at (usually root)
	 * @return the height of the tree
	 */
	private int getHeight(BSTNode N) {
		if (N == null)
			return 0;
		else {
			int leftHeight = getHeight(N.lLink);
			int rightHeight = getHeight(N.rLink);

			if (leftHeight > rightHeight) {
				return (leftHeight + 1);
			} else {
				return (rightHeight + 1);
			}
		}
	}
	
	
	/**
	 * Returns a linked list of the contents of the trees level
	 * @param N the node to start at
	 * @param level the level to work on 
	 * @return A LinkedList containing the level info
	 */
	private LinkedList<E> getLevel(BSTNode N ,int level, LinkedList toReturn)
    {
		
        if (N == null)
            return null;
        if (level == 1)
           	toReturn.insertAt(toReturn.getLength()+1, (E)N.key);
        else if (level > 1)
        {
        	toReturn.addAll(getLevel(N.lLink, level-1, toReturn));
        	toReturn.addAll(getLevel(N.rLink, level-1, toReturn));
        }
        return toReturn;
    }
	
	/**
	 * Returns a linked list of the contents of the trees level
	 * @param level The level you want to get the info from
	 * @return A LinkedList containing the level info
	 */
	public LinkedList<E> getLevel(int level) {
		LinkedList<E> list=new LinkedList<E>();
		return getLevel(root,level, list);
	}
	
	

	/**
	 * finds the height of the tree
	 * 
	 * @return The height of the tree
	 */
	public int getHeight() {
		return getHeight(root);
	}

	
	/**
	 * returns a visual representaiton of the tree
	 * @param N the node to start at 
	 * @param n a number for the offset of the values
	 * @return a visual representaiton of the tree
	 */
	private String displayTree(BSTNode<E> N) {
		String toReturn="";
		
		for(int i=0; i<this.getHeight();i++) { // goes through each layer
			LinkedList info = getLevel(i);
			String delimiter="";
			for(int j=0;j<this.getSize()/(i+2);j++) { //gets the right delimiter size
				delimiter+="\t";
			}
			
			while(info.getData(0)!=null) {
				toReturn+=delimiter;
				toReturn+=info.removeAt(0).toString();
			}
		}
		return toReturn;
	}
	
	/**
	 * returns a visual representaiton of the tree
	 * @return a visual representaiton of the tree
	 */
	public String displayTree() {
		return displayTree(root);
	}
}