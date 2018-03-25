from twilio.rest import Client as TwilioClient

twilio_sid = "<sid>"
twilio_auth_token = "<token>"
from_number = "<tel-number>" # Actually it should be a twilio number, see help on twilio.com.
to_number = "<tel-number>"

def snd_msg(sid, token, from_number, to_number,  msg_body):
    print("Send SMS from: {}, to: {}, body:\n\n{}".format(from_number, to_number, msg_body))
    client = TwilioClient(sid, token)
    msg = client.messages.create(to = to_number, from_ = from_number, body = msg_body)



snd_msg(twilio_sid, twilio_auth_token, from_number, to_number,  "hello,world")
