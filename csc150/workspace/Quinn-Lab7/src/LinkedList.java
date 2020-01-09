/**
 * Linked List is a collection of data nodes. All methods here relate to how one
 * can manipulate those nodes.
 * 
 * @author Xavier
 * @version 02/16/17
 * 
 *I affirm that I have carried out the attached academic endeavors with full academic honesty, in
 *accordance with the Union College Honor Code and the course syllabus.
 * 
 */
public class LinkedList {
	private int length; // number of nodes
	private ListNode firstNode; // pointer to first node

	public LinkedList() {
		length = 0;
		firstNode = null;
	}

	/**
	 * getter
	 * 
	 * @return number of nodes in the list
	 */
	public int getLength() {
		return length;
	}

	/**
	 * insert new Event at linked list's head
	 * 
	 * @param newData
	 *            the Event to be inserted
	 */
	public void insertAtHead(Event newData) {
		ListNode newnode = new ListNode(newData);
		if (length == 0) {
			firstNode = newnode;
		} else {
			newnode.next = firstNode;
			firstNode = newnode;
		}
		length++;
	}

	/**
	 * @return a string representation of the list and its contents.
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
	 * insert new Event into sorted position in LL
	 * 
	 * @param newData
	 *            the Event to insert
	 */
	public void insertSorted(Event newData) {
		ListNode nodeBefore = this.findNodeBefore(newData);
		if (nodeBefore == null) // if there isn't a node that should go before
								// newNode
			insertAtHead(newData);
		else
			insertAfter(nodeBefore, newData);
	}

	/**
	 * Given a new event to be inserted in the list, finds the correct position
	 * for it.
	 * 
	 * @param newData
	 *            an event to be inserted in the list
	 * 
	 * @return a pointer to the node in the linked list that will immediately
	 *         precede newData once newData gets inserted. Returns null if no
	 *         such node exists (which means newData goes first).
	 */
	private ListNode findNodeBefore(Event newData) {
		ListNode n;
		n=firstNode;
		
		if(length<=0) 
			return null;
		
		if(newData.compareTo(n.data)<0) { //If earlier
			return null;
		}
		
		while(n.next!=null && newData.compareTo(n.data)<0) {
			n=n.next;
		}
		return n;

	}
	
	
	


	/**
	 * Given an event to insert and a pointer to the node that should come
	 * before it, insert the new event after nodeBefore.
	 * 
	 * @param nodeBefore
	 *            the node (already in the list) that should immediately precede
	 *            the node with newData in it
	 * @param newData
	 *            the event to be inserted after nodeBefore
	 */
	private void insertAfter(ListNode nodeBefore, Event newData) {

		ListNode newNode = new ListNode(newData);
		
		newNode.next=nodeBefore.next;
		nodeBefore.next=newNode;
		
		length++;
		
	}
	
	/**
	 * Searches through for the specified event and returns it, if it doesn't exist it returns null
	 * @param month The month for the event
	 * @param day The Day for the event
	 * @param year The year for the event
	 * @param startingTime The starting time for the event
	 * @return The event or null
	 */
	public Event search(int month, int day, int year, int startingTime) {
		ListNode n;
		n=firstNode;
		Event tmp = new Event("tmp", year, month, day, startingTime, 1200);
		
		while(n.data.compareTo(tmp)!=0) {
			if(n.next==null) {
				return null;
			}
			n=n.next;
		}
		return n.data;
		
		
		
	}

}