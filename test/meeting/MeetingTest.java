package meeting;

import com.sun.jdi.request.DuplicateRequestException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.time.LocalTime;
import java.util.InvalidPropertiesFormatException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class MeetingTest {
    Meeting meeting;

    @BeforeEach
    void init() {
        this.meeting = new Meeting(0, 0);
    }

    @AfterEach
    void teardown() {
        this.meeting = null;
    }

    @Test
    @DisplayName("Can set meeting time")
    void canSetMeetingTime() throws InvalidPropertiesFormatException {
        int hour = 9;
        int minute = 0;
        LocalTime expectedStartTime = LocalTime.of(hour, minute);

        meeting.setMeetingTime(hour, minute);
        assertEquals(expectedStartTime, meeting.getMeetingStartTime());
    }

    @Test
    @DisplayName("Can't set meeting beyond top of the hour")
    void cantSetMeetingBeyondTopOfHour() {
        int hour = 9;
        int minute = 10;
        assertThrows(InvalidPropertiesFormatException.class, () -> meeting.setMeetingTime(hour, minute));
    }

    @Test
    @DisplayName("Can add attendee to meeting")
    void canAddAttendeeToMeeting() {
        String name = "Jake Shields";
        String email = "j.shields@company.com";
        Person attendee = new Person(name, email);
        meeting.addAttendee(attendee);
        assertEquals(attendee, meeting.getAttendeeByEmail(email));
    }

    @Test
    @DisplayName("Can't add duplicate attendees")
    void cantAddDuplicateAttendees() {
        String name = "Jake Shields";
        String email = "j.shields@company.com";
        Person attendee = new Person(name, email);
        meeting.addAttendee(attendee);

        assertThrows(DuplicateRequestException.class, () -> meeting.addAttendee(attendee));
    }
}
