/**
 * 
 * A generic binary search tree that holds comparables.
 * 
 *  *          * I affirm that I have carried out the attached academic endeavors
 *          with full academic honesty, in accordance with the Union College
 *          Honor Code and the course syllabus.
 *          
 * @author Xavier
 * @version 03.02.17
 */
public class BinarySearchTree<E extends Comparable<E>> {
	private BSTNode<E> root;

	/**
	 * Default constructor for BST, holds nothing until something is inserted
	 */
	public BinarySearchTree() {
		root=null;
	}
	
	/**
	 * Removes the node with given comparable
	 * @param toRemove the comparable to remove
	 * @param node The node to start at, usually root
	 * @return the new tree that has had the given data removed

	 */
    private BSTNode remove(E toRemove, BSTNode node){
        if( node == null ) {
            return node;
        }
        
        if( node.key.compareTo(toRemove) > 0 ) {
            node.lLink = remove( toRemove, node.lLink );
        }
        
        else if( node.key.compareTo(toRemove) < 0 ) {
            node.rLink = remove( toRemove, node.rLink);
        }
        
        else if(node.lLink != null && node.rLink != null )
        {
            node.key = inOrderSuccessor(node).key;
            node.rLink = remove( (E)node.key, node.rLink );
        }
        else {
        	
            if( node.lLink != null ) {
            	node=node.lLink;
            }
            else {
            	node=node.rLink; //Can be the leaf becoming null
            }
        }

        return node;
    }
	
    /**
     * Finds the in order successor of the given node
     * @param start The node to start at
     * @return The in order successor
     */
	private BSTNode inOrderSuccessor(BSTNode start) {
		return findMin(start.rLink);
	}
	
	/**
	 * Finds the minimum node after the given value
	 * @param start The node to start looking at
	 * @return The smallest node
	 */
	private BSTNode findMin(BSTNode start) {
		if(start.lLink==null) {
			return start;
		}
		else {
			return findMin(start.lLink);
		}
	}
    
	
	/**
	 * Removes the node with the given data and restructures the tree 
	 * accordingly
	 * @param toRemove The data you want to remove.
	 *
	 */
	public void remove(E toRemove) {
		root = remove(toRemove, root);
	}

	
	/**
	 * Inserts information into the tree
	 * @param newValue
	 */
	public void insert(E newValue) {
		root = insert(root,newValue);
	}
	
	/**
	 * Inserts the given data into the tree
	 * 
	 * @param whatever you want to put in the tree
	 * @return The new tree with the inserted node
	 **/
	private BSTNode insert(BSTNode<E> node, E data) {
		if (node == null) {
			return new BSTNode<E>(data);           
		} 
		else if ( data.compareTo(node.key) >= 0 ) { 
			node.rLink = insert(node.rLink, data);
			return node;
		} 

		else {
			node.lLink = insert(node.lLink, data); 
			return node;   
		} 
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
	 * Searches for some data and returns its data, or null if the node is not
	 * found
	 * 
	 * @param data
	 *            Whatever you want to find
	 * @return The data, or null
	 */
	public E search(E data) {
		BSTNode toReturn = search(root, data);
		if(toReturn!=null) {
			return (E)toReturn.key;
		}
		else {
			return null;
		}
	}

	
	
	/**
	 * Counts how many nodes there are in the tree
	 * 
	 * @return Number of nodes in the tree
	 */
	public int getSize() {
		return getSize(root);
	}
	
	/**
	 * 
	 * @param N The node to start at, usually root
	 * @return The size of the tree
	 */
	private int getSize(BSTNode N) {
		if (N==null) {
			return 0;
		}
		else {
			return getSize(N.lLink) + getSize(N.rLink) + 1;
		}
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
			toReturn += "  " + N.key.toString() + " ";
			toReturn += stringNodes(N.rLink);
			toReturn += ")";

		}
		return toReturn;
	}
	
	
	
	/**
	 * Returns just the values in the tree in order as a string (no visualization of the trees shape)
	 * @return just the values in the tree in order as a string
	 */
	public String sortedString() {
		return sortedString(root);
	}
	
	
	/**
	 * Returns just the values in the tree in order as a string (no visualization of the trees shape)
	 * @param N The node to start at
	 * @return just the values in the tree in order as a string
	 */
	private String sortedString(BSTNode<E> N) {
		String toReturn = "";
		if (N != null) {
			toReturn += sortedString(N.lLink);
			toReturn += N.key.toString() + "\n";
			toReturn += sortedString(N.rLink);
		}
		return toReturn;
	}
	
}