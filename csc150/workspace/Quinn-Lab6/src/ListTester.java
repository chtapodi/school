/** Tester for the Event and LinkedList classes
 * 
 * @author <em>Xavier Quinn</em>, Matt Anderson, Aaron Cass, and Chris Fernandes
 * @version 2/08/17
 */
public class ListTester {
    public static final boolean VERBOSE = true;
    
    public static void main(String[] args)
    {
    	Testing.setVerbose(true);
    	System.out.println("Starting Tests");
	
    	testConstructor();
    	testCompareTo();
    	testRemoveHead();
    	testInsertAtTail();
    	testSearch();

	
    	System.out.println("Tests Complete");
    }
    
    private static void testConstructor()
    {

    	Testing.testSection("Event Constructor test");
	
    	Event e1 = new Event("book club", 2012, 2, 24, 1000, 1200);
    	Testing.assertEquals("Non-default constructor", "book club  2/24/2012  1000-1200", e1.toString());
    }
    
    private static void testCompareTo()
    {
    	Testing.testSection("Event compareTo tests");
	
    	Event later = new Event("book club", 2012, 2, 24, 1000, 1200);
    	Event earlier = new Event("chess club", 2011, 2, 24, 900, 1000);
    	Testing.assertEquals("different years", 1, later.compareTo(earlier));
    	Testing.assertEquals("different years", -1, earlier.compareTo(later));
    	
    	Event laterToo= new Event("Boyscouts", 2012, 2, 24, 1000, 1200);
    	Testing.assertEquals("Exact same", 0, later.compareTo(laterToo));
    	
    	Event earlierMonth= new Event("Pi Contest", 2012, 1, 24, 1000, 1200);
    	Testing.assertEquals("Same, but a month before", -1, earlierMonth.compareTo(later));
    	Testing.assertEquals("Same, but a month after", 1, later.compareTo(earlierMonth));
    	
    	
    	Event earlierDay= new Event("Jazz Cub", 2012, 2, 23, 1000, 1200);
    	Testing.assertEquals("Same, but a day before", -1, earlierDay.compareTo(later));
    	Testing.assertEquals("Same, but a day after", 1, later.compareTo(earlierDay));
    	
    	Event earlierTime= new Event("Girlscouts", 2012, 2, 24, 900, 1100);
    	Testing.assertEquals("Same, but an hour before", -1, earlierTime.compareTo(later));
    	Testing.assertEquals("Same, but an hour after", 1, later.compareTo(earlierTime));
    	

    }
    
    
    private static void testRemoveHead() {
    	
    	Event later = new Event("Plant club", 2012, 2, 24, 1000, 1200);
    	Event earlier = new Event("Congress meeting", 2011, 2, 24, 900, 1000);
    	
    	LinkedList list = new LinkedList();
    	
    	Testing.assertEquals("Checks if head has been removed", null, list.removeHead());
    	
    	list.insertAtHead(earlier);
    	list.insertAtHead(later);
    	list.insertAtHead(earlier);
    	
    	Testing.assertEquals("Checks if head has been removed on filled list", "@Congress meeting  2/24/2011  900-1000@", "@" + list.removeHead() + "@");

    	Testing.assertEquals("Checks if head has been removed on filled list", "(Plant club  2/24/2012  1000-1200,\nCongress meeting  2/24/2011  900-1000)", list.toString());
    	Testing.assertEquals("Checks if size is correct", 2, list.getLength());
    	
    	list.removeHead();
    	list.removeHead();
    	
    	Testing.assertEquals("Checks if head has been removed on an empty list", "()", list.toString());
    	
    	
    	
    	
    }
    
    private static void testInsertAtTail() {
    	
    	Testing.testSection("Testing insertAtTail");
    	
    	LinkedList list = new LinkedList();
    	
    	Event later = new Event("Sumo Wrestling", 2012, 2, 24, 1000, 1200);
    	Event earlier = new Event("Baking Contest", 2011, 2, 24, 900, 1000);
    	
    	list.insertAtTail(later);

    	Testing.assertEquals("Checks if insertAtTail worked on an empty list", "(Sumo Wrestling  2/24/2012  1000-1200)", list.toString());

    	list.insertAtTail(earlier);
    	Testing.assertEquals("Checks if insertAtTail worked on an empty list", "(Sumo Wrestling  2/24/2012  1000-1200,\nBaking Contest  2/24/2011  900-1000)", list.toString());
    	
    	Testing.assertEquals("Checks if size is correct", 2, list.getLength());
    	
    }
    
    
    private static void testSearch() {
    	
    	LinkedList list = new LinkedList();
    	
    	Event later = new Event("Dueling Club", 2012, 2, 24, 1000, 1200);
    	Event earlier = new Event("Russian Roulette", 2011, 2, 24, 900, 1000);
    	Event middle = new Event("Knitting", 2016, 5, 12, 2300, 2330);
    	
    	Testing.assertEquals("Checks if search works on empty list", -1, list.search("Dueling Club"));
    	
    	list.insertAtHead(later);
    	list.insertAtHead(earlier);
    	list.insertAtHead(middle);
    	
    	
    	Testing.assertEquals("Checks if search works on populated list", 1000, list.search("Dueling Club"));
    	Testing.assertEquals("Checks if search works on populated list", 2300, list.search("Knitting"));
    	Testing.assertEquals("Checks if search works on populated list", 900, list.search("Russian Roulette"));
    	Testing.assertEquals("Checks if false search works on populated list", -1, list.search("Steves birthday"));
    	


    	
    }
    
    // Add more testing methods here when you get to Step 2 of the lab 

}