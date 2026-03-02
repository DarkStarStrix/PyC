use crate::errors::VortexError;

pub fn send_message(message: &str) -> Result<(), VortexError> {
    println!("Sending message via Mesocarp: {}", message);
    Ok(())
}
