/** 
 * 
 * The reminder is a a LogEntry that contains something you should remember to do
 * 
 * @author Chris Fernandes and Xavier
 * @version 2/20/12
 */
public class Reminder implements LogEntry
{
    private String msg;
    private String date;

    /**
     * Creates a reminder
     *
     * @param msg The text of the reminder
     * @param date The date of the reminder (in mm/dd/yyyy format)
     */
    public Reminder(String msg, String date)
    {
	   this.msg = msg;
	   this.date = date;
    }

    /**
     * @return the Reminder as a String
     */
    public String toString()
    {
	   return msg + " "  + date;
    }

    /**
     * @return the month of the reminder
     */
    public int getMonth()
    {
	   return getDateParts()[0];    
    }

    /**
     * @return the day of the reminder
     */
    public int getDay()
    {
	   return getDateParts()[1];    
    }
    
    
    public int getYear()
    {
	   return getDateParts()[2];    
    }

    /**
     * Parses the date string to determine the 3 integer values for
     * month, day, and year. Returns -1 if there is no date
     *
     * @return an array of 3 ints in the order {month, day, year}
     */
    private int[] getDateParts()
    {
	   int[] intParts = new int[3];
	   
	   if(date==null) {
		   
		   for (int i = 0; i < intParts.length; i++) {
		       intParts[i] = -1;
		   } 
		  return intParts;
	   }
	   
	   String[] stringParts = date.split("/");

	   for (int i = 0; i < stringParts.length; i++) {
	       intParts[i] = Integer.valueOf(stringParts[i]).intValue();
	   }

	   return intParts;
    }
}