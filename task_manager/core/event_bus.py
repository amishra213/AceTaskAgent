"""
Event Bus - Centralized event management for TaskManager

This module implements a pub-sub event bus for decoupled, event-driven architecture.
Enables reactive workflows where components subscribe to events and react automatically.

Features:
- Type-safe event publishing and subscription
- Priority-based event handling
- Event filtering and routing
- Async and sync event handlers
- Event history and replay
- Dead letter queue for failed events

Based on: INTERFACE_STANDARDS.md v1.0
"""

import asyncio
import logging
from typing import Callable, Optional, Any, Dict, List
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field
import traceback
import uuid

from task_manager.models.messages import SystemEvent, create_system_event
from task_manager.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EventSubscription:
    """Represents a subscription to an event type."""
    subscription_id: str
    event_type: str
    handler: Callable[[SystemEvent], Any]
    filter_func: Optional[Callable[[SystemEvent], bool]] = None
    priority: int = 5  # 1=highest, 10=lowest
    is_async: bool = False
    max_retries: int = 3
    retry_count: int = 0
    active: bool = True
    subscriber_name: str = "unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EventRecord:
    """Record of an event that was published."""
    event: SystemEvent
    published_at: str
    handlers_notified: List[str]
    handlers_succeeded: List[str]
    handlers_failed: List[str]
    processing_time_ms: int


class EventBus:
    """
    Centralized event bus for pub-sub messaging pattern.
    
    Usage:
        # Initialize
        event_bus = EventBus()
        
        # Subscribe to events
        def on_task_complete(event: SystemEvent):
            print(f"Task {event['payload']['task_id']} completed!")
        
        event_bus.subscribe(
            event_type="task_completed",
            handler=on_task_complete,
            subscriber_name="master_planner"
        )
        
        # Publish events
        event = create_system_event(
            event_type="task_completed",
            event_category="task_lifecycle",
            source_agent="pdf_agent",
            payload={"task_id": "task_1.2.3"}
        )
        event_bus.publish(event)
        
        # Async support
        async def on_ocr_results(event: SystemEvent):
            await process_ocr_data(event['payload'])
        
        event_bus.subscribe(
            event_type="ocr_results_ready",
            handler=on_ocr_results,
            subscriber_name="synthesis_agent",
            is_async=True
        )
    """
    
    def __init__(self, enable_history: bool = True, history_max_size: int = 1000):
        """
        Initialize the event bus.
        
        Args:
            enable_history: Whether to keep event history
            history_max_size: Maximum number of events to keep in history
        """
        self.subscriptions: Dict[str, List[EventSubscription]] = defaultdict(list)
        self.wildcard_subscriptions: List[EventSubscription] = []
        
        # Event history
        self.enable_history = enable_history
        self.history_max_size = history_max_size
        self.event_history: List[EventRecord] = []
        
        # Dead letter queue for failed events
        self.dead_letter_queue: List[tuple[SystemEvent, str]] = []
        
        # Statistics
        self.stats = {
            "events_published": 0,
            "events_delivered": 0,
            "events_failed": 0,
            "handlers_executed": 0,
            "handlers_failed": 0
        }
        
        logger.info("Event Bus initialized")
    
    def subscribe(
        self,
        event_type: str,
        handler: Callable[[SystemEvent], Any],
        subscriber_name: str = "unknown",
        filter_func: Optional[Callable[[SystemEvent], bool]] = None,
        priority: int = 5,
        is_async: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        Subscribe to an event type.
        
        Args:
            event_type: Type of event to subscribe to (or "*" for all events)
            handler: Function to call when event occurs
            subscriber_name: Name of the subscriber (for logging)
            filter_func: Optional filter function (return True to receive event)
            priority: Handler priority (1=highest, 10=lowest)
            is_async: Whether handler is async
            max_retries: Maximum retry attempts for failed handlers
        
        Returns:
            subscription_id: Unique subscription ID (for unsubscribing)
        """
        subscription = EventSubscription(
            subscription_id=str(uuid.uuid4()),
            event_type=event_type,
            handler=handler,
            filter_func=filter_func,
            priority=priority,
            is_async=is_async,
            max_retries=max_retries,
            subscriber_name=subscriber_name
        )
        
        if event_type == "*":
            self.wildcard_subscriptions.append(subscription)
            logger.debug(f"Wildcard subscription added: {subscriber_name}")
        else:
            self.subscriptions[event_type].append(subscription)
            # Sort by priority
            self.subscriptions[event_type].sort(key=lambda s: s.priority)
            logger.debug(f"Subscription added: {subscriber_name} → {event_type}")
        
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from events.
        
        Args:
            subscription_id: ID returned from subscribe()
        
        Returns:
            True if subscription was found and removed
        """
        # Check wildcard subscriptions
        for i, sub in enumerate(self.wildcard_subscriptions):
            if sub.subscription_id == subscription_id:
                self.wildcard_subscriptions.pop(i)
                logger.debug(f"Wildcard subscription removed: {sub.subscriber_name}")
                return True
        
        # Check typed subscriptions
        for event_type, subs in self.subscriptions.items():
            for i, sub in enumerate(subs):
                if sub.subscription_id == subscription_id:
                    subs.pop(i)
                    logger.debug(f"Subscription removed: {sub.subscriber_name} → {event_type}")
                    return True
        
        return False
    
    def publish(self, event: SystemEvent, async_mode: bool = False):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
            async_mode: Whether to handle async subscribers
        """
        start_time = datetime.now()
        event_type = event['event_type']
        
        logger.info(f"📢 EVENT PUBLISHED: {event_type} from {event['source_agent']}")
        logger.debug(f"Event payload: {event['payload']}")
        
        self.stats["events_published"] += 1
        
        # Check if event should propagate
        if not event.get('propagate', True):
            logger.debug(f"Event {event_type} marked as non-propagating, skipping handlers")
            return
        
        # Get all relevant subscriptions
        typed_subs = self.subscriptions.get(event_type, [])
        all_subs = typed_subs + self.wildcard_subscriptions
        
        # Track which handlers were notified/succeeded/failed
        handlers_notified = []
        handlers_succeeded = []
        handlers_failed = []
        
        # Notify all subscribers
        for subscription in all_subs:
            if not subscription.active:
                continue
            
            # Apply filter if specified
            if subscription.filter_func and not subscription.filter_func(event):
                logger.debug(f"Event filtered out for {subscription.subscriber_name}")
                continue
            
            handlers_notified.append(subscription.subscriber_name)
            
            try:
                if subscription.is_async and async_mode:
                    # Handle async subscribers
                    asyncio.create_task(self._async_handler_wrapper(event, subscription))
                else:
                    # Handle sync subscribers
                    self._execute_handler(event, subscription)
                
                handlers_succeeded.append(subscription.subscriber_name)
                self.stats["handlers_executed"] += 1
                
            except Exception as e:
                handlers_failed.append(subscription.subscriber_name)
                self.stats["handlers_failed"] += 1
                logger.error(
                    f"Handler {subscription.subscriber_name} failed for event {event_type}: {e}"
                )
                logger.debug(traceback.format_exc())
                
                # Retry logic
                if subscription.retry_count < subscription.max_retries:
                    subscription.retry_count += 1
                    logger.info(
                        f"Retrying handler {subscription.subscriber_name} "
                        f"(attempt {subscription.retry_count}/{subscription.max_retries})"
                    )
                    try:
                        self._execute_handler(event, subscription)
                        handlers_succeeded.append(f"{subscription.subscriber_name}_retry")
                    except Exception as retry_error:
                        logger.error(f"Retry failed: {retry_error}")
                        # Add to dead letter queue after max retries
                        if subscription.retry_count >= subscription.max_retries:
                            self.dead_letter_queue.append((event, str(e)))
                            logger.warning(f"Event added to dead letter queue: {event_type}")
        
        # Update statistics
        if handlers_succeeded:
            self.stats["events_delivered"] += 1
        if handlers_failed:
            self.stats["events_failed"] += 1
        
        # Record in history
        if self.enable_history:
            processing_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            record = EventRecord(
                event=event,
                published_at=start_time.isoformat(),
                handlers_notified=handlers_notified,
                handlers_succeeded=handlers_succeeded,
                handlers_failed=handlers_failed,
                processing_time_ms=processing_time_ms
            )
            self.event_history.append(record)
            
            # Trim history if needed
            if len(self.event_history) > self.history_max_size:
                self.event_history = self.event_history[-self.history_max_size:]
        
        logger.info(
            f"✅ Event {event_type} delivered to {len(handlers_succeeded)}/{len(handlers_notified)} handlers"
        )
    
    def _execute_handler(self, event: SystemEvent, subscription: EventSubscription):
        """Execute a single event handler."""
        logger.debug(f"Executing handler: {subscription.subscriber_name}")
        subscription.handler(event)
        subscription.retry_count = 0  # Reset retry count on success
    
    async def _async_handler_wrapper(self, event: SystemEvent, subscription: EventSubscription):
        """Wrapper for async event handlers."""
        try:
            logger.debug(f"Executing async handler: {subscription.subscriber_name}")
            await subscription.handler(event)
            subscription.retry_count = 0
            self.stats["handlers_executed"] += 1
        except Exception as e:
            self.stats["handlers_failed"] += 1
            logger.error(
                f"Async handler {subscription.subscriber_name} failed: {e}"
            )
            logger.debug(traceback.format_exc())
            
            # Retry logic for async handlers
            if subscription.retry_count < subscription.max_retries:
                subscription.retry_count += 1
                logger.info(f"Retrying async handler (attempt {subscription.retry_count})")
                await asyncio.sleep(1)  # Brief delay before retry
                await self._async_handler_wrapper(event, subscription)
    
    def get_subscriptions(self, event_type: Optional[str] = None) -> List[EventSubscription]:
        """
        Get all subscriptions, optionally filtered by event type.
        
        Args:
            event_type: Optional event type to filter by
        
        Returns:
            List of subscriptions
        """
        if event_type:
            return self.subscriptions.get(event_type, [])
        else:
            all_subs = []
            for subs in self.subscriptions.values():
                all_subs.extend(subs)
            all_subs.extend(self.wildcard_subscriptions)
            return all_subs
    
    def get_event_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[EventRecord]:
        """
        Get event history, optionally filtered.
        
        Args:
            event_type: Optional event type to filter by
            limit: Maximum number of records to return
        
        Returns:
            List of event records (most recent first)
        """
        if not self.enable_history:
            return []
        
        history = self.event_history[::-1]  # Reverse for most recent first
        
        if event_type:
            history = [r for r in history if r.event['event_type'] == event_type]
        
        return history[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            **self.stats,
            "active_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "wildcard_subscriptions": len(self.wildcard_subscriptions),
            "event_types_registered": len(self.subscriptions),
            "dead_letter_queue_size": len(self.dead_letter_queue),
            "history_size": len(self.event_history)
        }
    
    def clear_dead_letter_queue(self):
        """Clear the dead letter queue."""
        cleared = len(self.dead_letter_queue)
        self.dead_letter_queue.clear()
        logger.info(f"Dead letter queue cleared ({cleared} events)")
    
    def replay_event(self, event_id: str):
        """
        Replay a specific event from history.
        
        Args:
            event_id: ID of event to replay
        """
        for record in self.event_history:
            if record.event['event_id'] == event_id:
                logger.info(f"Replaying event: {event_id}")
                self.publish(record.event)
                return
        
        logger.warning(f"Event {event_id} not found in history")


# ============================================================================
# GLOBAL EVENT BUS INSTANCE
# ============================================================================

# Global singleton instance
_global_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """
    Get the global event bus instance (singleton).
    
    Returns:
        Global EventBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = EventBus()
    return _global_event_bus


def publish_event(event: SystemEvent):
    """
    Convenience function to publish event to global bus.
    
    Args:
        event: Event to publish
    """
    get_event_bus().publish(event)


def subscribe_to_event(
    event_type: str,
    handler: Callable[[SystemEvent], Any],
    subscriber_name: str = "unknown",
    **kwargs
) -> str:
    """
    Convenience function to subscribe to global bus.
    
    Args:
        event_type: Event type to subscribe to
        handler: Handler function
        subscriber_name: Name of subscriber
        **kwargs: Additional subscription options
    
    Returns:
        Subscription ID
    """
    return get_event_bus().subscribe(
        event_type=event_type,
        handler=handler,
        subscriber_name=subscriber_name,
        **kwargs
    )


# ============================================================================
# DECORATORS
# ============================================================================

def event_handler(event_type: str, subscriber_name: Optional[str] = None, **kwargs):
    """
    Decorator to register a function as an event handler.
    
    Usage:
        @event_handler("task_completed", subscriber_name="my_agent")
        def on_task_complete(event: SystemEvent):
            print(f"Task completed: {event['payload']}")
    """
    def decorator(func: Callable[[SystemEvent], Any]):
        name = subscriber_name or func.__name__
        get_event_bus().subscribe(
            event_type=event_type,
            handler=func,
            subscriber_name=name,
            **kwargs
        )
        return func
    return decorator
