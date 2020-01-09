/**
 * @author xavier
 */
public class BSTNode<E extends Comparable<E>>
{
    public E key;
    public BSTNode lLink;  
    public BSTNode rLink;
    
    /** Non-default constructor
     * 
     * @param input is whatever data you want stored in the node.
     */
    public BSTNode(E input)
    {
        this.key = input;
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
    