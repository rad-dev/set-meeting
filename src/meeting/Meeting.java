package meeting;

import com.sun.jdi.request.DuplicateRequestException;

import java.time.LocalTime;
import java.util.ArrayList;
import java.util.InvalidPropertiesFormatException;
import java.util.List;

public class Meeting {
    LocalTime startTime;
    LocalTime endTime;
    List<Person> attendees;

    public Meeting(int hour, int minute) {
        try {
            setMeetingTime(hour, minute);
        } catch (InvalidPropertiesFormatException e) {
            e.printStackTrace();
        }
        attendees = new ArrayList<>();
    }

    public void addAttendee(Person attendee) {
        if (!attendees.contains(attendee)) {
            attendees.add(attendee);
        } else {
            throw new DuplicateRequestException("User " + attendee.getName() + " is already in the meeting.");
        }
    }

    public Person getAttendeeByEmail(String email) {
        return this.attendees.stream().filter(attendee ->
                email.equals(attendee.getEmail())).findFirst().orElse(null);
    }

    public void setMeetingTime(int hour, int minute) throws InvalidPropertiesFormatException {
        if (isValidTime(hour, minute)) {
            this.startTime = LocalTime.of(hour, minute);
        } else {
            throw new InvalidPropertiesFormatException("Invalid minute value of " + minute);
        }
        int possibleEndTime = hour + 1;
        this.endTime = possibleEndTime > 23 ? LocalTime.of(0, minute) :
                LocalTime.of(possibleEndTime, minute);
    }

    public LocalTime getMeetingStartTime() {
        return this.startTime;
    }

    private boolean isValidTime(int hour, int minute) {
        return minute == 0;
    }
}
