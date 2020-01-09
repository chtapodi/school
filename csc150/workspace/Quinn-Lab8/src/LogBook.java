/** 
 * An ADT that stores events and reminders on a month by month basis
 *
 * @author Xavier
 * @version 2/23/17
 */
public class LogBook
{

	private int month;
	private int year;
	private LogEntry[] book;
	
    /**
     * Creates a logbook for the specific month
     *
     * @param month The month that it is (mm)
     * @param Year the year that is is (yyyy)
     */
    public LogBook(int month, int year) {
	   this.month = month;
	   this.year = year;
	   book = new LogEntry[31];
	   
    }
    
    /**
     * Inserts the given LogEntry into the LogBook
     * @param toInsert the LogEntry you want to insert
     * @return True if inserted, false if not
     */
    public boolean insertEntry(LogEntry toInsert) {
    	if(toInsert.getMonth()!=this.month || toInsert.getYear()!=this.year 
    			|| book[toInsert.getDay()]!=null || toInsert.getDay()<1 
    			|| toInsert.getDay()>31 || toInsert.getMonth()<1
    			|| toInsert.getMonth()>12) { //If not a valid entry
    		return false;
    	}
    	else {
    		book[toInsert.getDay()]=toInsert;
    	}
    	
    	return true;

    }
    
    
    /**
     * Returns the log entry at the given date, or null if one doesnt exist
     * @param date The date to get the entry from
     * @return the entry at the given date if it exists, otherwise null
     */
    public LogEntry getEntry(int date) {
    	return book[date];
    }
    
    public String toString() {
    	String toReturn="LogBook of " + this.month + "/" + this.year +"\n";
    	for(int i=0;i<this.book.length;i++) {
    		
    		if(this.getEntry(i)!=null) {
    			toReturn+=getEntry(i).toString() + "\n";
    		}
    	}
    	
    	
    	return toReturn;
    }

 
}