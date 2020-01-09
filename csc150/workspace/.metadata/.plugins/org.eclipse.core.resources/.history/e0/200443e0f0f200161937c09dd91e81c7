/**
 * Linked List is a collection of data nodes. All methods here relate to how one
 * can manipulate those nodes.
 * 
 * @author Xavier
 * @version 02.09.17
 * 
 *  * I affirm that I have carried out the attached academic endeavors with full academic honesty, in
 * accordance with the Union College Honor Code and the course syllabus.
 */
public class LinkedList {
	private int length; // number of nodes
	private ListNode firstNode; // pointer to first node

	public LinkedList() {
		length = 0;
		firstNode = null;
	}

	/**
	 * insert new Event at linked list's head
	 * 
	 * @param newData the Event to be inserted
	 */
	public void insertAtHead(Event newData) {
		ListNode newnode = new ListNode(newData);
		if (getLength() == 0) {
			firstNode = newnode;
		} else {
			newnode.next = firstNode;
			firstNode = newnode;
		}
		length++;
	}
	
	
	/**
	 * Inserts the Event at the end of the linked list
	 * @param newData The Event to insert
	 */
	public void insertAtTail(Event newData) {
		ListNode newNode = new ListNode(newData);

		if (getLength() == 0) {
			firstNode = newNode;
		} 
		else {

			ListNode n;
			n = firstNode;
			
			while (n.next != null) {
				n = n.next;
				System.out.println("CheckPoint");
			}
			
			n.next = newNode;
			
		}
		length++;
	}

	/**
	 * Removes the lists first node and returns the Data from the removed Event
	 * 
	 * @return Data from the removed Event
	 */
	public Event removeHead() {
		if (getLength() == 0) {
			return null;
		} else {
			Event tmpData = firstNode.data;
			firstNode = firstNode.next;
			length--;
			return tmpData;
		}
	}
	
	/**
	 * Searches through the linked list for a string
	 * @param searchString What the Events name is
	 * @return The start time of the event
	 */
	public int search(String searchString) {
		ListNode n;
		n = firstNode;
		while (n!=null && n.data.getName() != searchString) {
			n = n.next;
		}
		if(n!=null) {
			return n.data.getStart();
		}
		return -1;
		
	}

	/**
	 * Turn entire chain into a string
	 * 
	 * @return return linked list as printable string of format
	 *         (string,\nstring,\nstring)
	 */
	public String toString() {
		String toReturn = "(";
		ListNode n;
		n = firstNode;
		while (n != null) {
			toReturn = toReturn + n; // call node's toString automatically
			n = n.next;
			if (n != null) {
				toReturn = toReturn + ",\n";
			}
		}
		toReturn = toReturn + ")";
		return toReturn;
	}

	/**
	 * getter for number of nodes in the linked list
	 * 
	 * @return length of LL
	 */
	public int getLength() {
		return length;
	}
}