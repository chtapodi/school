/**
 * ListNode is a building block for a linked list of data items
 * 
 * This is the only class where I'll let you use public instance variables.
 * It's so we can reference information in the nodes using cascading dot
 * notation, like 
 *          N.next.data instead of 
 *          N.getNext().getData()
 * 
 * @author C. Fernandes and G. Marten and Xavier
 * @version 02.09.17
 */
public class ListNode<T>
{
    public T data;
    public ListNode next;   // pointer to next node
    
    /** Non-default constructor
     * 
     * @param String a reservation you want stored in this node
     */
    public ListNode(T String)
    {
        this.data = String;
        this.next = null;
    }
    
    // if you say "System.out.println(N)" where N is a ListNode, the
    // compiler will call this method automatically to print the contents
    // of the node.  It's the same as saying "System.out.println(N.toString())"
    public String toString()
    {
    	if(this.data!=null) 
    		return data.toString();
        return null;  // call the toString() method in String class
    }
}
    