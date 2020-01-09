/**
 * A node for a Binary Search Tree
 * Holds comparables
 * @author xavier
 */
public class BSTNode<E extends Comparable<E>>
{
    public E key;
    public BSTNode lLink;  
    public BSTNode rLink;

    
    /** Non-default constructor
     * Makes a node for an ADT to hold
     * has two child pointers
     * 
     * @param data is the comparable you want stored in the node.
     */
    public BSTNode(E data)
    {
        this.key = data;
        this.lLink = null;
        this.rLink = null;

    }
    

    /**
     * @return Returns the toString of the data in this node.
     */
    public String toString()
    {
    	if(this.key!=null) 
    		return key.toString();
        return null;
    }
    

    
}
    