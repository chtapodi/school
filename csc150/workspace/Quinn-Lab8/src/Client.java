/**
 * Tester for the LogBook, Event, and Reminder classes
 * 
 * @author <em>Xavier</em>, Aaron Cass, and Chris Fernandes
 * @version 5/14/15
 */
public class Client {
	public static final boolean VERBOSE = true;

	public static void main(String[] args) {
		Testing.setVerbose(VERBOSE);
		Testing.startTests();

		testReminder();
		testEvent();
		testInsertionAndToString();
		testGetEvent();
		changeTimes();

		Testing.finishTests();
	}

	private static void testReminder() {

		Reminder r1 = new Reminder("Buy Eggs", "02/24/2017");

		Testing.assertEquals("Tests getDay", 24, r1.getDay());

		Testing.assertEquals("Tests getMonth", 2, r1.getMonth());

		Testing.assertEquals("Tests getMonth", 2017, r1.getYear());

		Testing.assertEquals("Tests toString", "Buy Eggs 02/24/2017",
				r1.toString());

		Reminder r2 = new Reminder(null, null);

		Testing.assertEquals("Tests getDay", -1, r2.getDay());

		Testing.assertEquals("Tests getMonth", -1, r2.getMonth());

		Testing.assertEquals("Tests getMonth", -1, r2.getYear());

		Testing.assertEquals("Tests toString", "null null", r2.toString());

	}

	private static void testEvent() {

		Event e1 = new Event("test", 2017, 2, 24, 1100, 1200);

		Testing.assertEquals("Tests getDay", 24, e1.getDay());

		Testing.assertEquals("Tests getMonth", 2, e1.getMonth());

		Testing.assertEquals("Tests getMonth", 2017, e1.getYear());

		Testing.assertEquals("Tests toString", "test  2/24/2017  1100-1200",
				e1.toString());

		Testing.assertEquals("Tests getStart", 1100, e1.getStart());

		Testing.assertEquals("Tests getEnd", 1200, e1.getEnd());

		Reminder r2 = new Reminder(null, null);

		Testing.assertEquals("Tests getDay", -1, r2.getDay());

		Testing.assertEquals("Tests getMonth", -1, r2.getMonth());

		Testing.assertEquals("Tests getMonth", -1, r2.getYear());

		Testing.assertEquals("Tests toString", "null null", r2.toString());

	}

	private static void testInsertionAndToString() {

		Event e1 = new Event("chess", 2017, 2, 25, 1900, 1930);
		Event e2 = new Event("boy scouts", 2017, 2, 23, 900, 1000);
		Event e3 = new Event("girl scouts", 2017, 2, 24, 900, 1000);
		Event e4 = new Event("Steves birthday", 2017, 2, 24, 910, 1100);

		Reminder r1 = new Reminder("Buy Goat", "02/26/2017");
		Reminder r2 = new Reminder("Buy farm", "02/20/2016");
		Reminder r3 = new Reminder("Buy food", "03/27/2016");
		Reminder r4 = new Reminder("Apocolypse", "13/27/2016");
		Reminder r5 = new Reminder("Begining of time and space", "03/-1/2016");

		LogBook book = new LogBook(2, 2017);

		Testing.assertEquals("Tests Normal insertion of LogEntry", true,
				book.insertEntry(e1));
		Testing.assertEquals("Tests toString of normal insertion",
				"LogBook of 2/2017\nchess  2/25/2017  1900-1930\n",
				book.toString());

		Testing.assertEquals("Tests Normal insertion of LogEntry", true,
				book.insertEntry(e2));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\nchess  2/25/2017  1900-1930\n",
				book.toString());

		Testing.assertEquals("Tests Normal insertion of LogEntry", true,
				book.insertEntry(e3));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\n",
				book.toString());

		Testing.assertEquals("Tests Normal insertion of LogEntry", true,
				book.insertEntry(r1));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

		Testing.assertEquals("Tests insertion of LogEntry where one exists",
				false, book.insertEntry(e4));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

		Testing.assertEquals("Tests insertion of LogEntry of different year",
				false, book.insertEntry(r2));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

		Testing.assertEquals("Tests insertion of LogEntry of different month",
				false, book.insertEntry(r3));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

		Testing.assertEquals("Tests insertion of LogEntry of impossible month",
				false, book.insertEntry(r4));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

		Testing.assertEquals("Tests insertion of LogEntry of impossible month",
				false, book.insertEntry(r5));
		Testing.assertEquals(
				"Tests toString of normal insertion",
				"LogBook of 2/2017\nboy scouts  2/23/2017  900-1000\ngirl scouts  2/24/2017  900-1000\nchess  2/25/2017  1900-1930\nBuy Goat 02/26/2017\n",
				book.toString());

	}

	private static void testGetEvent() {

		Event e1 = new Event("chess", 2017, 2, 25, 1900, 1930);

		Reminder r1 = new Reminder("Buy Goat", "02/26/2017");

		LogBook book = new LogBook(2, 2017);

		book.insertEntry(e1);
		book.insertEntry(r1);

		Testing.assertEquals("Tests getEntry", e1, book.getEntry(25));

		Testing.assertEquals("Tests getEntry", r1, book.getEntry(26));

		Testing.assertEquals("Tests getEntry", null, book.getEntry(27));

	}

	private static void changeTimes() {

		LogBook book = populate(14);

		Reminder r1 = new Reminder("Buy Goat", "02/26/2017");
		Reminder r2 = new Reminder("Buy farm", "02/20/2016");
		book.insertEntry(r1);
		book.insertEntry(r2);

		for (int i = 10; i <= 15; i++) {
			if (book.getEntry(i) instanceof Event) {
				((Event) book.getEntry(i)).setStart(1600);
				((Event) book.getEntry(i)).setEnd(1730);
			}

		}

		Testing.assertEquals("Tests the changing times thing",
				"LogBook of 2/2017\nEvent 1  2/1/2017  900-1000\n"
						+ "Event 2  2/2/2017  900-1000\n"
						+ "Event 3  2/3/2017  900-1000\n"
						+ "Event 4  2/4/2017  900-1000\n"
						+ "Event 5  2/5/2017  900-1000\n"
						+ "Event 6  2/6/2017  900-1000\n"
						+ "Event 7  2/7/2017  900-1000\n"
						+ "Event 8  2/8/2017  900-1000\n"
						+ "Event 9  2/9/2017  900-1000\n"
						+ "Event 10  2/10/2017  1600-1730\n"
						+ "Event 11  2/11/2017  1600-1730\n"
						+ "Event 12  2/12/2017  1600-1730\n"
						+ "Event 13  2/13/2017  1600-1730\n"
						+ "Event 14  2/14/2017  1600-1730\n"
						+ "Buy Goat 02/26/2017\n", book.toString());

	}

	/**
	 * Populates a LogBook for testing purposes. Starts at the first day of the
	 * month and adds until the endDate
	 * 
	 * @param endDate
	 *            the last day you want envents on
	 * @return A filled logbook
	 */
	private static LogBook populate(int endDate) {
		LogBook toReturn = new LogBook(02, 2017);
		for (int i = 1; i <= endDate; i++) {
			Event e = new Event("Event " + i, 2017, 2, i, 900, 1000);
			toReturn.insertEntry(e);
		}

		return toReturn;

	}

}