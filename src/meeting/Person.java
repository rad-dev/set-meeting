package meeting;

public class Person {
  public static final int NINE_AM = 9;
  public static final int ELEVEN_AM = 11;
  public static final int ONE_PM = 1;
  public static final int FIVE_PM = 5;
  private String name;
  private  String email;
  private  boolean[] dailyMeetingSchedule;

  public Person(String name, String email) {
    this.name = name;
    this.email = email;
    this.dailyMeetingSchedule =  new boolean[8];              //assuming 9am to 6pm with 12 to 1pm as 1-hour lunch break and no overtimes
    /*
    *   [0] [01] [02] [03] [04] [05] [06] [07]
    *   [9] [10] [11] [01] [02] [03] [04] [05]
    * */
  }

  public String getEmail() {
    return email;
  }

  public void setEmail(String email) {
    this.email = email;
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public boolean setMeeting(int time) {
    if (isMorning(time)){
      time -= 9;
      this.dailyMeetingSchedule [time] = true;
      return true;
    } else if (isAfternoon(time)) {
      time += 2;
      this.dailyMeetingSchedule[time] = true;
      return true;
    }
    return false;
  }

  private boolean isAfternoon(int time) {
    return (time >= ONE_PM) && (time <= FIVE_PM);
  }

  private boolean isMorning(int time) {
    return (time >= NINE_AM) && (time <= ELEVEN_AM);
  }

  public boolean equals(Person anotherPerson) {
    return this.name.equals(anotherPerson.getName()) &&
            this.email.equals(anotherPerson.getEmail());
  }

  public String printMeetings() {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < 3; i++) {
      sb.append("Time: " + (i+9) + "am");
      checkSchedule(sb, i);
    }
    for (int i = 3; i < this.dailyMeetingSchedule.length; i++) {
      sb.append("Time: " + (i-2) + "pm");
      checkSchedule(sb, i);
    }
    return sb.toString();
  }

  private void checkSchedule(StringBuilder sb, int i) {
    if (this.dailyMeetingSchedule[i]) {
      sb.append(" - not available \n");
    } else {
      sb.append(" - available \n");
    }
  }
}
